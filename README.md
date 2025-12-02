# Cost-Sensitive Freeze-Thaw Bayesian Optimization for Efficient Hyperparameter Tuning
[![arXiv](https://img.shields.io/badge/arXiv-Read%20paper-b31b1b?style=flat&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2510.00492)

This repository contains the official codebase for our NeurIPS 2025 paper,  
**“Cost-Sensitive Freeze-Thaw Bayesian Optimization for Efficient Hyperparameter Tuning.”**

---

### Quick Start

```bash
conda create -n cfbo python=3.11
conda activate cfbo
pip install -r requirements.txt
```

---

### Data

Download the dataset from this [Google Drive link](https://drive.google.com/file/d/1SWhf1v6QGFDvYvPxXv7yVZiTfOCD0oAs/view?usp=drive_link) and unzip it into this repository.

---

### Learning-Curve (LC) Extrapolator

(Optional) Pretrain the LC extrapolator for transfer learning:

```bash
# BENCHMARK_NAME ∈ ["lcbench", "taskset", "pd1", "odbench"]
python train.py --benchmark_name BENCHMARK_NAME
```

Alternatively, download pretrained checkpoints from this [Google Drive link](https://drive.google.com/file/d/1SjCSx63qFgAOpKqyXu8rpYx9IH4BUTPb/view?usp=sharing) and unzip them into this repository.

---

### Cost-Sensitive Bayesian Optimization

We consider the following utility function: $U(b, \tilde{y}_b) = \tilde{y}_b - \alpha\left(\frac{b}{B}\right)^c$, where:

- $b$ denotes the currently consumed budget, and $\tilde{y}_b$ denotes the best performance observed up to budget $b$,
- **budget\_limit** ($B \in \mathbb{N}$): the maximum allowable optimization budget,
- **alpha** ($\alpha \in [0,1]$): the penalty coefficient for budget consumption ($\alpha = 0$ recovers conventional BO),
- **c** ($c > 0$): controls the curvature of the utility function (e.g., $c=1$ for linear, $c=2$ for quadratic, $c=0.5$ for square-root).



Run BO:

```bash
# DyHPO
# BENCHMARK_NAME ∈ ["lcbench", "taskset", "pd1", "odbench"]
python run_bo.py --algorithm dyhpo --benchmark_name BENCHMARK_NAME --alpha ALPHA --c C

# ifBO
python run_bo.py --algorithm ifbo --benchmark_name BENCHMARK_NAME --alpha ALPHA --c C

# CFBO without transfer learning
python run_bo.py --algorithm CFBO --benchmark_name BENCHMARK_NAME --alpha ALPHA --c C

# CFBO with transfer learning
python run_bo.py --algorithm CFBO --benchmark_name BENCHMARK_NAME --alpha ALPHA --c C \
    --model_ckpt ./checkpoints/BENCHMARK_NAME/model.pt
```

---

## Citation

```bibtex
@inproceedings{CFBO,
    title={Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning},
    author={Lee, Dong Bok and Zhang, Aoxuan Silvia and Kim, Byungjoo and Park, Junhyeon and Adriaensen, Steven and Lee, Juho and Hwang, Sung Ju and Lee, Hae Beom},
    booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/pdf?id=ZUb4JpNoJe}
}
```
