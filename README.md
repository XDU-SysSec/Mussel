# MUSSEL Artifact

This repository contains the code for the paper:

**"Hit The Bullseye On The First Shot: Improving LLMs Using Multi-Sample Self-Reward Feedback for Vulnerability Repair"**

## Overview

This artifact implements MUSSEL (MUlti-Sample SElf-reward Learning), a novel approach to improve LLMs' capabilities in vulnerability repair tasks. Instead of relying on multiple-round iterations or human feedback, our method generates multiple repair candidates in a single forward pass and employs a self-reward mechanism to provide more effective learning signals to the model.

## Key Features

- **Multi-Sample Self-Reward Feedback**: Generates multiple repair candidates and uses KL-divergence based weights to prioritize more informative examples
- **One-Shot Repair**: Performs vulnerability repair without requiring multiple iterations or human feedback
- **Improved Accuracy**: Demonstrates significant improvements over state-of-the-art models on vulnerability repair benchmarks

## Example Model
- Deepseek-MUSSEL
- https://drive.google.com/file/d/17DL0gI99qyoWhwoh88cweKx1zQvAhkGP/view?usp=sharing

## Core Implementation

The core implementation is located at `src/llmtuner/train/dpo/trainer.py` in the `dpo_multiple_loss_weighted` method. This method:

1. Processes multiple rejected candidates for each chosen sample
2. Computes KL-divergence between the chosen sample and each rejected candidate
3. Uses these divergences to assign weights to different learning signals
4. Incorporates length penalties to avoid excessive verbosity or brevity

## Acknowledgements

This research builds upon previous work in vulnerability repair and LLM optimization.

We would like to express our gratitude to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project, which provided the foundational framework and infrastructure that made this research possible. Their comprehensive and well-designed toolkit for LLM fine-tuning significantly accelerated our development process.

## Note:

Due to significant upgrades in several key dependencies and the integration of newer library versions, the training code
is currently undergoing active reorganization and refactoring.
We are working diligently to ensure compatibility with the latest ecosystem.

**Current Status:**

- The core training algorithm and methodology remain stable and functional
- We are updating dependency specifications
- Documentation for the training pipeline is being revised to reflect recent changes
- We expect to complete this reorganization in the near future

We appreciate your patience during this transition period.