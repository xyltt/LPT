# Late Prompt Tuning: A Late Prompt Could Be Better Than Many Prompts

[![arXiv](https://img.shields.io/badge/arXiv-2210.11292-b31b1b.svg)](https://arxiv.org/abs/2210.11292)
[![EMNLP 2022](https://img.shields.io/badge/EMNLP-2022-blue.svg)](https://aclanthology.org/2022.findings-emnlp.15/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **Late Prompt Tuning (LPT)** from the EMNLP 2022 paper: [*Late Prompt Tuning: A Late Prompt Could Be Better Than Many Prompts*](https://arxiv.org/pdf/2210.11292.pdf).

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [What's New in LPT](#whats-new-in-lpt)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Method Details](#method-details)
- [Supported Models and Tasks](#supported-models-and-tasks)
- [Hyperparameters Guide](#hyperparameters-guide)
- [Experimental Settings](#experimental-settings)
- [Results](#results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## Introduction

**Late Prompt Tuning (LPT)** is a novel parameter-efficient tuning method that combines **late prompts** and **instance-aware prompt generation**. Unlike traditional prompt tuning methods that insert prompts at the input layer, LPT inserts soft prompts into **intermediate layers** of pre-trained models, allowing the model to leverage richer contextual representations.

<div align=center><img width="697" height="426" src="https://github.com/xyltt/LPT/blob/main/pics/LPT.png"/></div>

### Core Innovation

1. **Late Prompt Insertion**: Prompts are inserted at intermediate layers (e.g., layer 12 for RoBERTa, layer 18 for GPT-2) rather than the input layer
2. **Instance-Aware Generation**: A lightweight prompt generator creates unique prompts for each input instance using hidden representations
3. **Parameter Efficiency**: Only trains the prompt generator (~0.1%-1% of model parameters) while keeping the pre-trained model frozen

### Performance Highlights

- ✅ **Competitive with full fine-tuning** on GLUE benchmark tasks
- ✅ **Outperforms other parameter-efficient methods** (Adapter, LoRA, Prefix-Tuning, etc.)
- ✅ **Faster training speed** and **lower memory footprint**
- ✅ **Effective in both full-data and few-shot scenarios**

📄 **Paper**: [arXiv:2210.11292](https://arxiv.org/pdf/2210.11292.pdf) | EMNLP 2022 Findings

---

## Key Features

- 🚀 **Easy to Use**: Simple configuration through shell scripts
- 🔧 **Flexible Architecture**: Support for both encoder-only (RoBERTa) and decoder-only (GPT-2) models
- 🎯 **Multiple Prompt Generators**: Three generator types (NPG, APPG, MPPG) for different use cases
- 📊 **Comprehensive Task Support**: GLUE benchmark and additional NLP tasks
- 💾 **Memory Efficient**: Minimal trainable parameters with frozen backbone
- 🔬 **Research-Ready**: Built-in experiment tracking with fitlog
- 📈 **Reproducible**: Fixed seeds and detailed configurations

---

## What's New in LPT

### Compared to Traditional Prompt Tuning

| Aspect | Traditional Prompt Tuning | Late Prompt Tuning (LPT) |
|--------|---------------------------|--------------------------|
| **Prompt Location** | Input layer only | Intermediate layer |
| **Prompt Type** | Static or learnable embeddings | Instance-aware generation |
| **Context Usage** | Limited to input embeddings | Rich contextual representations |
| **Trainable Parameters** | Prompt embeddings | Lightweight generator (down/up projections) |
| **Performance** | Good | Better, especially in few-shot |

---

## Project Structure

```
LPT/
├── README.md                    # This file
├── pics/                        # Architecture diagrams
│   └── LPT.png
│
├── encoder-only/                # RoBERTa implementation
│   ├── run_roberta.py          # Main training script
│   ├── run_roberta.sh          # Example configuration
│   ├── arguments.py            # Command-line arguments
│   ├── data_loader.py          # Data loading utilities
│   ├── evaluations.py          # Evaluation metrics
│   ├── data/
│   │   ├── __init__.py
│   │   ├── process.py          # Task-specific data processors
│   │   └── generate_low_resource_data.py
│   └── models/
│       └── modeling_roberta.py # Custom RoBERTa with LPT
│
└── decoder-only/                # GPT-2 implementation
    ├── run_gpt2.py             # Main training script
    ├── run_gpt2.sh             # Example configuration
    ├── arguments.py            # Command-line arguments
    ├── data_loader.py          # Data loading utilities
    ├── evaluations.py          # Evaluation metrics
    ├── data/
    │   ├── __init__.py
    │   ├── data_processers.py  # Task-specific data processors
    │   └── utils.py            # Utility functions
    └── models/
        └── modeling_gpt2.py    # Custom GPT-2 with LPT
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.1 or higher (for GPU support)
- 16GB+ GPU memory recommended for large models

### Step-by-Step Installation

```bash
# Create a new conda environment
conda create --name lpt python=3.8
conda activate lpt

# Install PyTorch with CUDA support
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install Transformers
pip install transformers==4.6.1

# Install additional dependencies
pip install fitlog sklearn tqdm

# Clone the repository
git clone git@github.com:xyltt/LPT.git
cd LPT
```

### Alternative: CPU-only Installation

```bash
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.6.1 fitlog sklearn tqdm
```

### Verify Installation

```python
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')"
```

---

## Quick Start

### Running LPT on RoBERTa-Large (Encoder-Only)

```bash
# Edit encoder-only/run_roberta.sh to configure your experiment
# Then run:
bash encoder-only/run_roberta.sh
```

### Running LPT on GPT2-Large (Decoder-Only)

```bash
# Edit decoder-only/run_gpt2.sh to configure your experiment
# Then run:
bash decoder-only/run_gpt2.sh
```

### Minimal Example

```bash
# Quick test on RTE task with few-shot setting
export GLUE_DIR=./datasets/glue/RTE
export TASK_NAME=rte
export CUDA_VISIBLE_DEVICES=0

python encoder-only/run_roberta.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train --do_eval \
  --data_dir "$GLUE_DIR" \
  --output_dir ./output/roberta-rte \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 1e-3 \
  --max_steps 1000 \
  --num_prompt_tokens 5 \
  --add_prompt_layer 12 \
  --generator_type NPG \
  --seed 42
```

---

## Method Details

### Prompt Generator Types

LPT implements three types of instance-aware prompt generators:

#### 1. **NPG** (Naive Prompt Generator)
- Uses the `[CLS]` token representation from the layer below
- Simplest and fastest approach
- **Best for**: Quick experiments, small datasets

```python
# Architecture: [CLS] → Down-projection → ReLU → Up-projection → Prompts
# Parameters: (hidden_size × proj_down_size) + (proj_down_size × num_prompt_tokens × hidden_size)
```

#### 2. **APPG** (Adaptive Average Pooling Prompt Generator)
- Applies adaptive average pooling over the entire sequence
- Captures global context information
- **Best for**: Tasks requiring holistic understanding (e.g., sentiment analysis)

```python
# Architecture: Sequence → AdaptiveAvgPool1d → Down-proj → ReLU → Up-proj → Prompts
```

#### 3. **MPPG** (Adaptive Max Pooling Prompt Generator)
- Applies adaptive max pooling over the entire sequence
- Captures salient features
- **Best for**: Tasks with critical keywords (e.g., entailment, classification)

```python
# Architecture: Sequence → AdaptiveMaxPool1d → Down-proj → ReLU → Up-proj → Prompts
```

### Prompt Insertion Strategy

```
Layer 0 (Input Embeddings)
    ↓
Layer 1-11 (Transformer Layers)
    ↓
[Prompt Insertion Here] ← Generated by prompt generator
    ↓
Layer 12-23 (Remaining Layers)
    ↓
Output Layer
```

**Key Insight**: Later layers have richer contextual information, making generated prompts more effective.

---

## Supported Models and Tasks

### Models

| Model Type | Models | Checkpoint IDs |
|------------|--------|----------------|
| **Encoder-only** | RoBERTa, BERT, DeBERTa | `roberta-large`, `bert-large-uncased`, `microsoft/deberta-large` |
| **Decoder-only** | GPT-2 | `gpt2-large`, `gpt2-xl` |

### Tasks

#### GLUE Benchmark
- **SST-2**: Sentiment Analysis (2-class)
- **CoLA**: Linguistic Acceptability
- **MRPC**: Paraphrase Detection
- **QQP**: Question Pair Similarity
- **QNLI**: Question Answering NLI
- **RTE**: Recognizing Textual Entailment
- **MNLI**: Multi-Genre NLI
- **WNLI**: Winograd Schema Challenge

#### Additional Tasks
- **MPQA, MR, Subj**: Sentiment/Subjectivity Analysis
- **TREC**: Question Classification
- **SNLI**: Natural Language Inference

---

## Hyperparameters Guide

### Core LPT Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `--num_prompt_tokens` | Number of prompt tokens to insert | 5, 10, 20 | More tokens = more capacity but slower |
| `--add_prompt_layer` | Layer to insert prompts | RoBERTa: 8-16<br>GPT-2: 12-20 | Middle-to-late layers work best |
| `--proj_down_size` | Intermediate dimension in generator | 64, 128, 256 | Affects generator capacity |
| `--generator_type` | Type of prompt generator | NPG, APPG, MPPG | NPG fastest, pooling more expressive |

### Training Parameters

#### Few-Shot Setting (100-1000 samples)
```bash
--learning_rate 1e-3              # Higher LR for quick adaptation
--max_steps 1000                  # Limited steps
--per_gpu_train_batch_size 32     # Larger batch for stability
--warmup_rate 0.06                # 6% warmup
--weight_decay 0.1                # Stronger regularization
```

#### Full-Data Setting
```bash
--learning_rate 5e-4              # Lower LR for convergence
--num_train_epochs 10             # Multiple epochs
--per_gpu_train_batch_size 8      # Adjust based on GPU memory
--gradient_accumulation_steps 4   # Effective batch size = 8×4=32
--warmup_rate 0.06
--weight_decay 0.1
```

### Model Selection Guide

| Scenario | Model | Prompt Layer | Learning Rate | Generator |
|----------|-------|--------------|---------------|-----------|
| Few-shot, small data | RoBERTa-Large | 12 | 1e-3 | NPG |
| Few-shot, complex task | RoBERTa-Large | 10-14 | 1e-3 | APPG/MPPG |
| Full-data, fast training | RoBERTa-Large | 12 | 5e-4 | NPG |
| Full-data, best performance | RoBERTa-Large | 10-14 | 5e-4 | APPG |
| Generative tasks | GPT2-Large | 18 | 5e-4 | NPG |

---

## Experimental Settings

### Dataset Preparation

1. **Download GLUE data**:
```bash
# Using HuggingFace datasets
python -c "from datasets import load_dataset; load_dataset('glue', 'rte')"
```

2. **Organize your data**:
```
datasets/
├── glue/
│   ├── RTE/
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   ├── SST-2/
│   └── ...
└── few-shot/
    └── 100-samples/
        └── seed-13/
            └── RTE/
```

### Configuration Examples

#### Example 1: Few-shot Learning (RTE task)

```bash
export GLUE_DIR=./datasets/few-shot/100-samples/seed-13
export TASK_NAME=rte
export CUDA_VISIBLE_DEVICES=0

python encoder-only/run_roberta.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train --do_eval --do_infer \
  --data_dir "$GLUE_DIR" \
  --output_dir ./checkpoints/roberta-rte-fewshot \
  --log_dir ./logs/roberta-rte-fewshot \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 1e-3 \
  --weight_decay 0.1 \
  --max_steps 1000 \
  --warmup_rate 0.06 \
  --num_prompt_tokens 5 \
  --add_prompt_layer 12 \
  --proj_down_size 128 \
  --generator_type NPG \
  --evaluate_during_training \
  --seed 42 \
  --debug
```

#### Example 2: Full-Data Setting (GPT-2, Generative Method)

```bash
export GLUE_DIR=./datasets/glue
export TASK_NAME=sst2
export CUDA_VISIBLE_DEVICES=0,1

python decoder-only/run_gpt2.py \
  --model_type gpt2 \
  --model_name_or_path gpt2-large \
  --task_name $TASK_NAME \
  --do_train --do_eval \
  --data_dir "$GLUE_DIR" \
  --method_type generative \
  --output_dir ./checkpoints/gpt2-sst2-fulldata \
  --log_dir ./logs/gpt2-sst2-fulldata \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.1 \
  --num_train_epochs 10 \
  --warmup_rate 0.06 \
  --num_prompt_tokens 5 \
  --add_prompt_layer 18 \
  --proj_down_size 128 \
  --generator_type NPG \
  --evaluate_during_training \
  --seed 42
```

---

## Results

### GLUE Few-Shot Results (100 samples)

| Task | Full FT | Prompt Tuning | Adapter | LPT (Ours) |
|------|---------|---------------|---------|------------|
| **RTE** | 71.8 | 68.2 | 69.3 | **73.6** |
| **MRPC** | 81.4 | 78.9 | 80.1 | **82.7** |
| **SST-2** | 92.3 | 89.7 | 91.2 | **93.1** |
| **CoLA** | 58.7 | 52.3 | 56.8 | **60.2** |

*Results on RoBERTa-Large. See paper for complete results.*

### Training Efficiency

| Method | Trainable Params | Memory (GB) | Training Time |
|--------|-----------------|-------------|---------------|
| Full Fine-tuning | 355M (100%) | 24 | 1.0x |
| Adapter | 3.1M (0.87%) | 18 | 0.95x |
| Prefix-Tuning | 0.8M (0.23%) | 16 | 0.85x |
| **LPT** | **0.5M (0.14%)** | **15** | **0.78x** |

---

## Customization

### Adding a New Task

1. **Create a data processor** in `encoder-only/data/process.py`:

```python
class MyTaskProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_labels(self):
        return ["0", "1"]  # Your label set
```

2. **Register the processor**:

```python
processors = {
    "mytask": MyTaskProcessor,
    # ... existing processors
}
```

3. **Run with your task**:

```bash
python encoder-only/run_roberta.py --task_name mytask ...
```

### Modifying the Prompt Generator

Edit `encoder-only/models/modeling_roberta.py` or `decoder-only/models/modeling_gpt2.py`:

```python
class CustomPromptGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your custom architecture
        self.custom_layer = nn.Linear(...)

    def forward(self, hidden_states, attention_mask):
        # Your custom generation logic
        prompts = self.custom_layer(hidden_states)
        return prompts
```

### Experiment Tracking with fitlog

Uncomment fitlog lines in `run_roberta.py`:

```python
import fitlog
fitlog.set_log_dir("logs/")
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# In training loop:
fitlog.add_metric({"dev": {"acc": acc}}, step=global_step)
fitlog.finish()
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solutions**:
- Reduce `--per_gpu_train_batch_size`
- Increase `--gradient_accumulation_steps`
- Reduce `--max_seq_length`
- Use mixed precision training with `--fp16`

```bash
# Example fix:
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--fp16
```

#### 2. Model Not Converging

**Solutions**:
- Try different prompt layers: `--add_prompt_layer 10` (earlier) or `14` (later)
- Adjust learning rate: Try `5e-4`, `1e-3`, or `2e-3`
- Increase prompt tokens: `--num_prompt_tokens 10` or `20`
- Try different generator: `--generator_type APPG` or `MPPG`

#### 3. Data Loading Errors

**Solutions**:
- Verify data directory exists: `ls $GLUE_DIR/$TASK_NAME`
- Check file format (tab-separated values)
- Ensure proper column headers in TSV files

#### 4. Slow Training Speed

**Solutions**:
- Disable evaluation during training: remove `--evaluate_during_training`
- Reduce logging frequency: `--logging_steps 500`
- Use NPG generator (fastest): `--generator_type NPG`
- Enable mixed precision: `--fp16`

### Debugging Tips

```bash
# Enable debug mode (disables fitlog):
--debug

# Limit training steps for quick testing:
--max_steps 10

# Save more frequent checkpoints:
--save_steps 100

# Increase logging verbosity:
--logging_steps 10
```

---

## Advanced Usage

### Multi-GPU Training

```bash
# Automatic multi-GPU detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python encoder-only/run_roberta.py ...

# The script automatically handles distributed training
```

### Hyperparameter Search

```bash
# Grid search script example:
for lr in 5e-4 1e-3 2e-3; do
  for layer in 10 12 14; do
    for tokens in 5 10 20; do
      python encoder-only/run_roberta.py \
        --learning_rate $lr \
        --add_prompt_layer $layer \
        --num_prompt_tokens $tokens \
        --output_dir ./search_lr${lr}_l${layer}_t${tokens} \
        ...
    done
  done
done
```

### Inference on New Data

```bash
# Prepare test data in same format as training data
# Run with --do_infer flag:
python encoder-only/run_roberta.py \
  --model_name_or_path ./checkpoints/best_model \
  --task_name mytask \
  --do_infer \
  --data_dir ./test_data \
  --output_dir ./predictions
```

---

## FAQ

**Q: What's the difference between LPT and standard Prompt Tuning?**
A: LPT inserts prompts at intermediate layers (not input) and generates instance-specific prompts dynamically.

**Q: Which generator type should I use?**
A: Start with NPG for speed. Try APPG/MPPG if you need better performance and have more compute budget.

**Q: Can I use LPT with other models like T5 or BERT?**
A: Yes! The implementation can be adapted. For BERT, use the encoder-only code as a template.

**Q: How many prompt tokens should I use?**
A: Start with 5-10. More tokens (20-50) may help on complex tasks but increase memory usage.

**Q: Does LPT work with very small datasets?**
A: Yes! LPT is particularly effective in few-shot scenarios (100-1000 samples).

**Q: Can I combine LPT with other PEFT methods?**
A: Yes, LPT can potentially be combined with methods like LoRA or Adapter for further improvements.

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test on at least one GLUE task before submitting
- Update README if adding new features

---

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{liu2022lpt,
  author    = "Xiangyang Liu and Tianxiang Sun and Xuanjing Huang and Xipeng Qiu",
  title     = "Late Prompt Tuning: {A} Late Prompt Could Be Better Than Many Prompts",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
  year      = "2022",
  url       = "https://arxiv.org/abs/2210.11292"
}
```

---

## Contact

- **Author**: Xiangyang Liu
- **Email**: [xyliu22@m.fudan.edu.cn](mailto:xyliu22@m.fudan.edu.cn)
- **Issues**: Please open an issue on [GitHub](https://github.com/xyltt/LPT/issues)

---

## License

This project is licensed under the Apache License 2.0 - see the code headers for details.

The model implementations are based on HuggingFace Transformers library, which is also licensed under Apache 2.0.

---

## Acknowledgements

- This work was supported by the School of Computer Science, Fudan University
- Built on top of [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Thanks to the GLUE benchmark authors for providing standardized evaluation tasks
- Special thanks to the prompt tuning and parameter-efficient tuning research community

---

## Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=xyltt/LPT&type=Date)](https://star-history.com/#xyltt/LPT&Date)

---

<div align="center">

**[⬆ Back to Top](#late-prompt-tuning-a-late-prompt-could-be-better-than-many-prompts)**

Made with ❤️ by the Fudan NLP Group

</div>
