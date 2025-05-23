# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning

![overall_framework](imgs/overall_framework.png)

[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://www.arxiv.org/pdf/2505.16270) &nbsp;&nbsp;


## **Introduction**
We introduce a novel Pilot-Copilot learning framework, named Transformer Copilot, that learns from a language model's "mistake log" during fine-tuning. By leveraging the Pilot model's learning signals, the Copilot model rectifies the Pilot model's output logits at inference time to improve generation accuracy. Our approach achieves notable gains on challenging reasoning tasks and downstream applications while preserving strong transferability and scalibility.

## **News**
- We've realeased our [paper](https://www.arxiv.org/pdf/2505.16270) on arXiv. Our code will be released soon! Stay Tuned!

## **Citation**
```
@misc{zou2025transformer,
  title        = {Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning},
  author       = {Jiaru Zou and Yikun Ban and Zihao Li and Yunzhe Qi and Ruizhong Qiu and Ling Yang and Jingrui He},
  year         = {2025},
  month        = {May},
  eprint       = {2505.16270},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2505.16270},
  doi          = {10.48550/arXiv.2505.16270}
}
```