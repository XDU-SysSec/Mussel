from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, List

import torch
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..utils import create_custom_optimzer, create_custom_scheduler
import torch.nn.functional as F


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self.beta = finetuning_args.dpo_beta
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.loss_type = finetuning_args.dpo_loss
        self.weight_factor = finetuning_args.weight_factor
        self.ftx_gamma = finetuning_args.dpo_ftx
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self.finetuning_args.multi_dpo:
            print("[MSR_DPO_LOG] run with msr dpo!")
            print("[MSR_DPO_LOG]self.beta:\n{}".format(self.beta))
            print("[MSR_DPO_LOG]self.weight_factor:\n{}".format(self.weight_factor))
            print("[MSR-DPO-INFO]enable_reciprocal: ", self.finetuning_args.enable_reciprocal)
        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def sft_loss(self, chosen_logits: "torch.FloatTensor", chosen_labels: "torch.LongTensor") -> "torch.Tensor":
        r"""
        Computes supervised cross-entropy loss of given labels under the given logits.

        Returns:
            A tensor of shape (batch_size,) containing the cross-entropy loss of each samples.
        """
        all_logps = self.get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
        return -all_logps

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under the given logits if loss_type != IPO.

        Otherwise the average log probabilities.
        """
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        all_logits: "torch.Tensor" = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            return_dict=True,
            use_cache=False,
        ).logits.to(torch.float32)

        all_logps = self.get_batch_logps(
            logits=all_logits,
            labels=batch_copied["labels"],
            average_log_prob=(self.loss_type == "ipo"),
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def concatenated_forward_multi(
            self, model: "PreTrainedModel", batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor], torch.FloatTensor, List[torch.FloatTensor]]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"], attention_mask=batch_copied["attention_mask"], return_dict=True
        ).logits.to(torch.float32)

        all_logps = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )
        items_per_batch = self.finetuning_args.multi_dpo_reject_data_num + 1
        batch_count = batch["input_ids"].size(0) // items_per_batch
        # Create chosen_logps and rejected_logps
        chosen_logps = all_logps[::items_per_batch]  # the first element of each batch
        rejected_logps = []
        chosen_logits = all_logits[::items_per_batch]  # similarly for logits
        rejected_logits = []

        for i in range(batch_count):
            # Add all elements except the first one for each batch
            start_idx = i * items_per_batch
            rejected_logps.append(all_logps[start_idx + 1:start_idx + items_per_batch])
            rejected_logits.append(all_logits[start_idx + 1:start_idx + items_per_batch])

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits
    def dpo_multiple_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps_list: List[torch.FloatTensor],
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps_list: List[torch.FloatTensor],
            batch: Dict[str, torch.Tensor],
            len_penalty_factor: float
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        losses_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        chosen_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        rejected_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        total_count = 0
        # get batch lens data
        seqs_len_all = batch['seqs_len'].to(policy_chosen_logps.device)
        items_per_batch = self.finetuning_args.multi_dpo_reject_data_num + 1
        batch_count = batch["input_ids"].size(0) // items_per_batch

        chosen_lens = seqs_len_all[::items_per_batch]  # the first element of each batch
        reject_lens = []
        for i in range(batch_count):
            start_idx = i * items_per_batch
            reject_lens.append(seqs_len_all[start_idx + 1:start_idx + items_per_batch])
        for i in range(len(policy_chosen_logps)):
            policy_chosen_logp = policy_chosen_logps[i]
            reference_chosen_logp = reference_chosen_logps[i]
            chosen_len = chosen_lens[i]
            for j in range(len(policy_rejected_logps_list[i])):
                policy_rejected_logp = policy_rejected_logps_list[i][j]
                reference_rejected_logp = reference_rejected_logps_list[i][j]
                reject_len = reject_lens[i][j]
                loss, chosen_reward, rejected_reward = self.dpo_loss(
                    policy_chosen_logp,
                    policy_rejected_logp,
                    reference_chosen_logp,
                    reference_rejected_logp
                )
                len_loss = self.google_length_penalty(reject_len, chosen_len, 0.6)
                losses_sum += loss
                losses_sum += len_loss * len_penalty_factor
                if self.finetuning_args.debug_dpo_loss:
                    print('[DEBUG_INFO] batch loss: ', loss)
                    print('[DEBUG_INFO] batch len_loss: ', len_loss * len_penalty_factor)
                chosen_rewards_sum += chosen_reward
                rejected_rewards_sum += rejected_reward
                total_count += 1

        average_losses = losses_sum / total_count
        average_chosen_rewards = chosen_rewards_sum / total_count
        average_rejected_rewards = rejected_rewards_sum / total_count

        return average_losses, average_chosen_rewards, average_rejected_rewards

    def kl_divergence(self, logits_p, logits_q):
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        kl = torch.sum(p * (log_p - log_q), dim=-1)
        return kl

    def dpo_multiple_loss_weighted_reciprocal(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps_list: List[torch.FloatTensor],
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps_list: List[torch.FloatTensor],
            policy_chosen_logits: torch.FloatTensor,
            policy_rejected_logits_list: List[torch.FloatTensor],
            batch: Dict[str, torch.Tensor],
            len_penalty_factor: float
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        losses_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        chosen_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        rejected_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        total_count = 0
        # get batch lens data
        seqs_len_all = batch['seqs_len'].to(policy_chosen_logps.device)
        items_per_batch = self.finetuning_args.multi_dpo_reject_data_num + 1
        batch_count = batch["input_ids"].size(0) // items_per_batch

        chosen_lens = seqs_len_all[::items_per_batch]  # the first element of each batch
        reject_lens = []

        for i in range(batch_count):
            start_idx = i * items_per_batch
            reject_lens.append(seqs_len_all[start_idx + 1:start_idx + items_per_batch])

        # 遍历batch
        for i in range(len(policy_chosen_logps)):
            policy_chosen_logp = policy_chosen_logps[i]
            reference_chosen_logp = reference_chosen_logps[i]
            policy_chosen_logit = policy_chosen_logits[i]
            policy_reject_logit_list = policy_rejected_logits_list[i]
            chosen_len = chosen_lens[i]
            if self.finetuning_args.debug_dpo_loss:
                print('[CookieInfo]kl_divergences_tensor: kl_weights = 1 / kl_divergences_tensor')
            with torch.no_grad():
                # Compute KL divergences and weights
                kl_divergences = [F.kl_div(F.log_softmax(policy_chosen_logit, dim=-1), F.softmax(rej_logit, dim=-1),
                                           reduction='batchmean')
                                  for rej_logit in policy_reject_logit_list]
                kl_divergences_tensor = torch.tensor(kl_divergences, device=policy_chosen_logps.device)
                kl_weights = 1 / kl_divergences_tensor  # Inverse to give smaller KL higher weight
                kl_weights /= kl_weights.sum()  # Normalizing

            # loop for inner msr-dpo loss
            for j in range(len(policy_rejected_logps_list[i])):
                policy_rejected_logp = policy_rejected_logps_list[i][j]
                reference_rejected_logp = reference_rejected_logps_list[i][j]
                reject_len = reject_lens[i][j]
                loss, chosen_reward, rejected_reward = self.dpo_loss(
                    policy_chosen_logp,
                    policy_rejected_logp,
                    reference_chosen_logp,
                    reference_rejected_logp
                )
                len_loss = self.google_length_penalty(reject_len, chosen_len, 0.6)
                final_loss = loss * 1.8 + len_loss * len_penalty_factor

                # Weighted sum of losses
                weighted_loss = kl_weights[j] * final_loss * self.weight_factor
                losses_sum += weighted_loss
                if self.finetuning_args.debug_dpo_loss:
                    print(
                        f'[DEBUG_INFO] Original Loss: {final_loss.item():.4f}, Weight: {kl_weights[j].item():.4f}, Weighted Loss: {weighted_loss.item():.4f}')
                chosen_rewards_sum += chosen_reward
                rejected_rewards_sum += rejected_reward
                total_count += 1

        average_losses = losses_sum / total_count
        average_chosen_rewards = chosen_rewards_sum / total_count
        average_rejected_rewards = rejected_rewards_sum / total_count

        return average_losses, average_chosen_rewards, average_rejected_rewards

    def dpo_multiple_loss_weighted(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps_list: List[torch.FloatTensor],
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps_list: List[torch.FloatTensor],
            policy_chosen_logits: torch.FloatTensor,
            policy_rejected_logits_list: List[torch.FloatTensor],
            batch: Dict[str, torch.Tensor],
            len_penalty_factor: float
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        losses_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        chosen_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        rejected_rewards_sum = torch.tensor(0.0, device=policy_chosen_logps.device)
        total_count = 0
        # get batch lens data
        seqs_len_all = batch['seqs_len'].to(policy_chosen_logps.device)
        items_per_batch = self.finetuning_args.multi_dpo_reject_data_num + 1
        batch_count = batch["input_ids"].size(0) // items_per_batch

        chosen_lens = seqs_len_all[::items_per_batch]  # the first element of each batch
        reject_lens = []

        for i in range(batch_count):
            start_idx = i * items_per_batch
            reject_lens.append(seqs_len_all[start_idx + 1:start_idx + items_per_batch])
        for i in range(len(policy_chosen_logps)):
            policy_chosen_logp = policy_chosen_logps[i]
            reference_chosen_logp = reference_chosen_logps[i]
            policy_chosen_logit = policy_chosen_logits[i]
            policy_reject_logit_list = policy_rejected_logits_list[i]
            chosen_len = chosen_lens[i]
            with torch.no_grad():
                # Compute KL divergences and weights
                kl_divergences = [F.kl_div(F.log_softmax(policy_chosen_logit, dim=-1), F.softmax(rej_logit, dim=-1),
                                           reduction='batchmean')
                                  for rej_logit in policy_reject_logit_list]
                kl_weights = torch.exp(torch.tensor(kl_divergences))  # Exponential to increase separation
                kl_weights /= kl_weights.sum()  # Normalizing

            # loop for inner msr-dpo loss
            for j in range(len(policy_rejected_logps_list[i])):
                policy_rejected_logp = policy_rejected_logps_list[i][j]
                reference_rejected_logp = reference_rejected_logps_list[i][j]
                reject_len = reject_lens[i][j]
                loss, chosen_reward, rejected_reward = self.dpo_loss(
                    policy_chosen_logp,
                    policy_rejected_logp,
                    reference_chosen_logp,
                    reference_rejected_logp
                )
                len_loss = self.google_length_penalty(reject_len, chosen_len, 0.6)
                final_loss = loss + len_loss * len_penalty_factor

                # Weighted sum of losses
                weighted_loss = kl_weights[j] * final_loss
                losses_sum += weighted_loss
                if self.finetuning_args.debug_dpo_loss:
                    print(
                        f'[DEBUG_INFO] Original Loss: {final_loss.item():.4f}, Weight: {kl_weights[j].item():.4f}, Weighted Loss: {weighted_loss.item():.4f}')
                chosen_rewards_sum += chosen_reward
                rejected_rewards_sum += rejected_reward
                total_count += 1

        average_losses = losses_sum / total_count
        average_chosen_rewards = chosen_rewards_sum / total_count
        average_rejected_rewards = rejected_rewards_sum / total_count

        return average_losses, average_chosen_rewards, average_rejected_rewards

    def google_length_penalty(self, aimed_seq_len, ref_seq_len, length_penalty_factor):
        """
        Calculate the length penalty term based on the Google length penalty formula.

        :param aimed_seq_len: The target sequence length tensor.
        :param ref_seq_len: The reference sequence length tensor.
        :param length_penalty_factor: A hyperparameter for controlling the strength of the penalty.
        :return: The length penalty term.
        """
        # Ensure that the length_penalty_factor is a float
        length_penalty_factor = float(length_penalty_factor)

        # Calculate the length penalty
        penalty = ((5 + aimed_seq_len) ** length_penalty_factor) / ((5 + ref_seq_len) ** length_penalty_factor)

        return penalty

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        if self.finetuning_args.multi_dpo:
            (
                policy_chosen_logps,
                policy_rejected_logps_list,
                policy_chosen_logits,
                policy_rejected_logits_list
            ) = self.concatenated_forward_multi(model, batch)
            if self.finetuning_args.debug_dpo:
                print(f"[MSR-DPO-INFO]concatenated_forward_multi 完成，当前GPU内存占用: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            with torch.no_grad():
                if self.ref_model is None:
                    ref_model = self.model
                    ref_context = self.accelerator.unwrap_model(self.model).disable_adapter()
                else:
                    ref_model = self.ref_model
                    ref_context = nullcontext()

                with ref_context:
                    (
                        ref_chosen_logps,
                        ref_rejected_logps_list,
                        _,
                        _
                    ) = self.concatenated_forward_multi(ref_model, batch)
            if self.finetuning_args.enable_msrdpo_weight:
                if self.finetuning_args.enable_reciprocal:
                    losses, chosen_rewards, rejected_rewards = self.dpo_multiple_loss_weighted_reciprocal(
                        policy_chosen_logps,
                        policy_rejected_logps_list,
                        ref_chosen_logps,
                        ref_rejected_logps_list,
                        policy_chosen_logits,
                        policy_rejected_logits_list,
                        batch,
                        self.finetuning_args.len_penalty_factor
                    )
                else:
                    losses, chosen_rewards, rejected_rewards = self.dpo_multiple_loss_weighted(
                        policy_chosen_logps,
                        policy_rejected_logps_list,
                        ref_chosen_logps,
                        ref_rejected_logps_list,
                        policy_chosen_logits,
                        policy_rejected_logits_list,
                        batch,
                        self.finetuning_args.len_penalty_factor
                    )
            else:
                losses, chosen_rewards, rejected_rewards = self.dpo_multiple_loss(
                    policy_chosen_logps,
                    policy_rejected_logps_list,
                    ref_chosen_logps,
                    ref_rejected_logps_list,
                    batch,
                    self.finetuning_args.len_penalty_factor
                )
            if self.finetuning_args.debug_dpo_loss:
                print("[DEBUG] losses:", losses)
                print("[DEBUG] chosen_rewards:", chosen_rewards)
                print("[DEBUG] rejected_rewards:", rejected_rewards)

            # Set prefix for train or eval
            if self.ftx_gamma > 1e-6:
                items_per_batch = self.finetuning_args.multi_dpo_reject_data_num + 1
                chosen_labels = batch["labels"][::items_per_batch]
                losses += self.ftx_gamma * self.sft_loss(policy_chosen_logits, chosen_labels).mean()
            if self.finetuning_args.debug_dpo_loss:
                print("[DEBUG] ftx_gamma_losses:", losses)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            self.log_msr_dpo_matrix(chosen_rewards, metrics, policy_chosen_logits, policy_chosen_logps, rejected_rewards,
                                reward_accuracies, train_eval)

        else:
            losses, metrics = self.native_dpo_compute_loss(batch, metrics, model, train_eval)

        torch.cuda.empty_cache()
        return losses, metrics

    def log_matrixs(self, chosen_rewards, logits_tuples, logps_tuples, metrics, rejected_rewards, reward_accuracies,
                    train_eval):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        policy_rejected_logps = torch.cat([t[1] for t in logps_tuples])
        policy_chosen_logps = torch.cat([t[0].unsqueeze(0) for t in logps_tuples])
        policy_rejected_logits = torch.cat([t[1] for t in logits_tuples])
        policy_chosen_logits = torch.cat([t[0].unsqueeze(0) for t in logits_tuples])
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()

    def log_msr_dpo_matrix(self, chosen_rewards, metrics, policy_chosen_logits, policy_chosen_logps, rejected_rewards,
                       reward_accuracies, train_eval):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

    def native_dpo_compute_loss(self, batch, metrics, model, train_eval):
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                ref_model = self.model
                ref_context = self.accelerator.unwrap_model(self.model).disable_adapter()
            else:
                ref_model = self.ref_model
                ref_context = nullcontext()

            with ref_context:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        if self.ftx_gamma > 1e-6:
            batch_size = batch["input_ids"].size(0) // 2
            chosen_labels, _ = batch["labels"].split(batch_size, dim=0)
            losses += self.ftx_gamma * self.sft_loss(policy_chosen_logits, chosen_labels)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
