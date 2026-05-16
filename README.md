# 🎮 Mobile Game Payer Prediction

> Binary classification of mobile game payers using deep neural networks 

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)

---

## Overview

This project implements a binary classification neural network to predict whether a mobile game player will make an in-app purchase (payer vs. non-payer)

---

## Project Structure

```
├── configs/                  # configuration files
│   ├── base.yaml             # Base model configuration
│   ├── best.yaml             # Best found hyperparameters of the sampler model
│   ├── sweep.yaml            # W&B hyperparameter sweep config
│   ├── sweep_focal_loss.yaml # 
│   ├── sweep_pos_weight.yaml # 
│   └── sweep_sampler.yaml    # 
│
├── scripts/                  # Scripts
│   ├── main.py               # Main script
│   ├── train.py              # Model training
│   ├── validate.py           # Model validation
│   ├── sweep.py              # Single sweep run
│   └── run_sweep.py          # Full W&B sweep
│
├── src/                      # Source code
│   ├── config.py             
│   ├── criterion.py          # Focal loss
│   ├── data.py               # Dataset and data loading
│   ├── data_exploration.py   # Feature distribution
│   ├── kfold.py              # K-fold helper
│   ├── model.py              # Model class
│   ├── trainer.py            # Trainer class
│   ├── results.py            
│   ├── utils.py              
│   └── explainability/       # Model explainability methods
│       ├── gradient.py       
│       ├── shapley.py        
│       ├── lime.py          
│       └── helper.py         
│
├── environment.yml           # Conda environment
└── README.md
```

---

## Requirements

- Python 3.10
- Conda (Miniconda or Anaconda)

Key libraries:

| Package | Version |
|---------|---------|
| PyTorch | 2.9.1 (CPU) |
| scikit-learn | 1.7.2 |
| SHAP | 0.48.0 |
| pandas | 2.2.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.3 |
| seaborn | 0.13.2 |
| scipy | 1.15.2 |
| wandb | 0.23.1 |


Full dependency list is in `environment.yml`.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/zarakarakaya/bc_thesis_payer_prediction.git
cd bc_thesis_payer_prediction
```

**2. Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate project
```

**3. Verify the installation**


## Usage

```bash
conda activate project
python -m scripts.run_sweep
```
