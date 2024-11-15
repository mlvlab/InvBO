# Inversion-based Latent Bayesian Optimization (InvBO)

Official PyTorch implementation of the "[Inversion-based Latent Bayesian Optimization](https://arxiv.org/pdf/2411.05330)".
(NeurIPS 2024)

> Jaewon Chu*, Jinyoung Park*, Seunghun Lee, Hyunwoo J. Kim†.

## Setup
- Clone repository
```
git clone https://github.com/mlvlab/InvBO.git
cd InvBO
```
- Install Environment
```
conda env create -f invbo.yml
conda activate invbo
pip install molsets==0.3.1 --no-deps
```

## Run Experiments
This repository provides InvBO applied to [CoBO](https://arxiv.org/pdf/2310.20258) [Lee et al., NeurIPS 2023] for small budget setting.
```
python exec.py --cuda 0 --task_idx [TASK]
```
Since we predefined the coefficients for VAE loss terms in `exec.py` provided by CoBO, the available tasks for [TASK] are:

| task_id |      Task Name      |
|---------|---------------------|
|  med2   | Median molecules 2  |
|  pdop   | Perindopril MPO     |
|  osmb   | Osimertinib MPO     |
|  adip   | Amlodipine MPO      |
|  zale   | Zaleplon MPO        |
|  valt   | Valsartan SMARTS    |
|  rano   | Ranolazine MPO      |

However, we can also run on the remaining Guacamol tasks when we define coefficients for VAE loss terms in `exec.py`:

| task_id |      Task Name      |
|---------|---------------------|
|  med1   | Median molecules 1  |
|  siga   | Sitagliptin MPO     |
|  dhop   | Deco Hop            |
|  shop   | Scaffold Hop        |
|  fexo   | Fexofenadine MPO    |

## Weights and Biases (wandb) tracking
You can track the optimization process using the wandb library.

You can use wandb tracking by simply setting `'--track_with_wandb', 'True'` and `'--wandb_entity', 'YOUR ENTITRY'` in `exec.py`.

## Acknowledgements
This repository is based on [CoBO](https://github.com/mlvlab/CoBO).
