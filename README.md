# Late Prompt Tuning: A Late Prompt Could Be Better Than Many Prompts

## Introduction

Late Prompt Tuning (LPT) is a prompt-based tuning method that combined late and instance-aware prompts. It inserts a soft prompt into an intermediate layer of pre-trained models. To further improve performance and take full advantage of these contextual hidden representations below the prompt insertion layer, LPT introduces a prompt generator to generate an independent prompt for each instance using the hidden representations. Its illustration is as follows. LPT can achieve competitive performance to full model tuning and other parameter-efficient tuning methods under both full-data and few-shot scenarios while possessing faster training speed and lower memory cost. More details are provided in our EMNLP paper [Late Prompt Tuning: A Late Prompt Could Be Better Than Many Prompts](https://arxiv.org/pdf/2210.11292.pdf).

<div align=center><img width="697" height="426" src="https://github.com/xyltt/LPT/blob/main/pics/LPT.png"/></div>

## Prepare your environment

The implementation of Late Prompt Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation based on `Pytorch`, `Transformers`. Optionally, you can use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name lpt python=3.8
conda activate lpt
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.6.1
pip install fitlog
pip install sklearn
git clone git@github.com:xyltt/LPT.git
cd LPT
```

## Using LPT

Now you can run LPT on RoBERTa-Large (encoder-only) and GPT2-Large (decoder-only) models with `run_roberta.sh` and `run_gpt2.sh`, respectively.

For running LPT on RoBERTa-Larg model:
```bash
bash enoder-only/run_roberta.sh 
```

For running LPT on GPT2-Larg model:
```bash
bash deoder-only/run_gpt2.sh 
```

For different experiment settings such as full-data setting and few-shot setting, you only need to adjust the dataset path and some hyperparamters like the number of training epochs. 

If you have any problems, raise an issue or contact [Xiangyang Liu](mailto:xyliu22@m.fudan.edu.cn) 

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{liu2022lpt,
  author    = "Xiangyang Liu and Tianxiang Sun and Xuanjing Huang and Xipeng Qiu",
  title     = "Late Prompt Tuning: {A} Late Prompt Could Be Better Than Many Prompts",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
  year      = "2022"
}
```
