# NanoChat-MoE

**NanoChat-MoE** is an end-to-end Mixture-of-Experts (MoE) training and evaluation pipeline built on top of **NanoChat**.  
It supports the full large language model lifecycle, including pretraining, mid-training, supervised fine-tuning (SFT), and reinforcement learning (RL), with a focus on training stability and scalable MoE design.
> **🚀 A minimal, hackable, and research-oriented MoE framework for studying and building scalable language models.**

## Overview

Mixture-of-Experts models enable scaling model capacity with nearly constant per-token compute, but in practice their training is often fragile due to router collapse, expert imbalance, and sensitivity to hyperparameters. 

NanoChat-MoE addresses these challenges by integrating sparse MoE layers into NanoChat with minimal architectural changes, while explicitly modeling MoE-specific auxiliary losses and diagnostics.

The resulting framework provides a full-stack MoE training pipeline, covering tokenization, training, evaluation, and analysis. It supports all major LLM training stages, incorporates sparse MoE layers with top-k routing and capacity constraints, and stabilizes optimization through **load-balancing loss** and **router z-loss regularization**. Evaluation is unified through **LM-Eval-Harness**, and all experiments are fully reproducible, enabling systematic scaling studies across model depth, width, and expert count.

## Model Architecture

NanoChat-MoE extends the standard NanoChat Transformer by selectively replacing MLP blocks with sparse Mixture-of-Experts (MoE) layers.

In each MoE layer, a lightweight router predicts expert assignment logits for every token. A top-k routing strategy dispatches tokens to a subset of MLP experts, while capacity constraints prevent individual experts from being overloaded.

To stabilize training and avoid expert collapse, we introduce two MoE-specific auxiliary losses:
- Load-balancing loss, which encourages uniform utilization across experts
- Router z-loss, which regularizes the magnitude of router logits

These auxiliary losses are aggregated across all MoE layers and jointly optimized with the primary language modeling objective, leading to significantly more stable training behavior.


## Quick start

Boot up a new 8XA800 GPU box from your favorite provider,  and start the training pipeline by running:
```
bash speedrun_moe.sh
```

The script runs the complete pipeline end-to-end, including pretraining, mid-training, fine-tuning, and evaluation.

If you are interested in the details of each training stage, we encourage you to inspect the script directly. It contains inline comments and explanations for every phase of the pipeline.

## Training Data

For each training phase, we align the dataset configuration with NanoChat, ensuring fair and controlled comparisons.
| Phase            | Training datasets | 
|---------------------|:--------:|
| Pretrain                | OpenWebText  | 
| Mid-train               | SmolTalk (460K), MMLU-auxiliary (100K), GSM8K-main (8K), CustomJSON-Identity (1K ×2 epochs), SimpleSpelling (200K), SpellingBee (80K)|
| SFT    | ARC-Easy (2.3K), ARC-Challenge (1.1K), GSM8K-main (8K), SmolTalk (10K), CustomJSON-Identity (1K), SimpleSpelling (300), SpellingBee (300)| 
| RL      | GSM8K |

## Training Dynamics

During pretraining, the optimization process typically produces a loss curve similar to the following:

<img src="assets/d6_pretrain.png" width="80%">

### Evaluation

We integrate **LM-Eval-Harness** for systematic evaluation across training phases. 

Below we report results for a 6-layer, 86M-A65M-parameter NanoChat-MoE model:

| Metric              | Pretrain | Mid-train | SFT    | RL|
|---------------------|:--------:|:---------:|:------:|:------:|
| Core                | 0.0836   | 0.0812    | 0.0706 | -- |
| MMLU                | 0.2296   | 0.2305    | 0.2424 |--  |
| GSM8K (flexible)    | --       | 0.0190    | 0.0205 | 0.0303 |
| GSM8K (strict)      | --       | 0.0053    | 0.0061 | 0.0212 |

The _Core_  benchmark aggregates ARC-Easy, ARC-Challenge, HellaSwag, PIQA, and Winogrande, using a normalized average score for fair cross-task comparison.

## Repository Structure

```text
nanochat-MoE/
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py   # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── configurator.py             # A superior alternative to argparse
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer with MoE layers
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── rustbpe                         # Custom Rust BPE tokenizer trainer
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # see for why this even exists
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun_moe.sh                 # Train a 6-layer 86M A65M NanochatMoE model
│
├── lm_eval.md                      # Tutorial for evaluation via LM-Eval-Harness
├── tools/lm_eval                   # Evaluation scripts
│
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
│   └── test_rustbpe.py
└── uv.lock
```

## Acknowledgements

We thank **Kexian Tang** and **Jinghan Li** for their guidance on code alignment, as well as their valuable suggestions on model training and scaling.  
We also thank **Kexian Tang** and **Shaowen Wang** for their advice on topic selection.  
We are grateful to **Professor Lyu** for continuous guidance throughout the project.  
Finally, we acknowledge the computing resources provided by **Volcano Engine (ByteDance)**, including access to A800 GPUs.


## Citation

If you use NanoChat-MoE in your research, please cite:
```
@misc{nanochat-moe,
  author={Duan, Peiqi and Wang, Jiani and Li, Muheng},
  title={NanoChat-MoE: An End-to-End Mixture-of-Experts Training Pipeline},
  year={2026},
  publisher = {GitHub},
  url = {https://github.com/Tangkexian/nanochat-MoE}
}
```