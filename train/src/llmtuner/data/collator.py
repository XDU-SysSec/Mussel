from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq

from llmtuner.hparams import FinetuningArguments


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None, label_pad_token_id=None, finetuning_args=None):
        super().__init__(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                         label_pad_token_id=label_pad_token_id)
        self.finetuning_args = finetuning_args

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        Masks out the input ids except for the responses.
        """
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        sequence_lengths = []
        for feature in features:
            # Process chosen_ids
            prompt_len, answer_len = len(feature["prompt_ids"]), len(feature["chosen_ids"])
            concatenated_features.append(
                {
                    "input_ids": feature["prompt_ids"] + feature["chosen_ids"],
                    "attention_mask": [1] * (prompt_len + answer_len),
                }
            )
            label_positions.append((prompt_len, answer_len))
            sequence_lengths.append(answer_len)

            # Process rejected_ids
            if self.finetuning_args.multi_dpo:
                # Process each rejected_ids in rejected_ids_list, if exists
                for rejected_ids in feature.get("rejected_ids_list", []):
                    prompt_len, answer_len = len(feature["prompt_ids"]), len(rejected_ids)
                    concatenated_features.append(
                        {
                            "input_ids": feature["prompt_ids"] + rejected_ids,
                            "attention_mask": [1] * (prompt_len + answer_len),
                        }
                    )
                    label_positions.append((prompt_len, answer_len))
                    sequence_lengths.append(answer_len)
            else:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature["rejected_ids"])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature["rejected_ids"],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = super().__call__(concatenated_features)
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        if self.finetuning_args.multi_dpo:
            batch["seqs_len"] = torch.tensor(sequence_lengths)
        if self.finetuning_args.multi_dpo and self.finetuning_args.debug_dpo:
            # Decode and print input_ids
            decoded_input_ids = [self.tokenizer.decode(ids) for ids in batch["input_ids"]]
            batch_size = len(features)
            separator = "-" * 40
            for i, decoded_input in enumerate(decoded_input_ids):
                if i > 0 and i % batch_size == 0:
                    print(separator)
                print(f"\nBatch Index {i // batch_size}, Data Index {i % batch_size}: {decoded_input}")
        return batch
