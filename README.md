# BIF


## Requirements

    python 3.7
    torch-1.4.0 
    matplotlib~=3.1.0
    argparse~=1.4.0
    scipy~=1.4.1
    autodp~=0.1
    scikit-learn~=0.21.2
    torchvision~=0.4.2
    pandas~=0.25.3
    seaborn~=0.10.0
    xgboost~=0.90
    sdgym~=0.1.0
    backpack-for-pytorch~=1.1.0
    
## Running the experiments
 

### I. Synthetic tabular datasets:

```
cd publish_bif_tab/code
python bif.py --dataset xor
```

Datasets: xor, subtact, orange_skin, nonlinear_additive, syn4, syn5, syn6, credit, adult_short, intrusion

The output is MCC (Mathews correlation coefficient).

### II. Real-world tabular datasets

Please download credit dataset folder and place it in `publish_bif_tab/data/`
https://www.dropbox.com/sh/ulzz7pca1wwgj6e/AACe5cNveQW_HH0TXGUb9Gnua?dl=0
The other datasets are in the Git repository


```
cd publish_bif_tab/code
python bif.py --dataset adult_short
```

Datasets: credit, adult_short, intrusion

The output is classification accuracy for k top features. To change the number of features selected specify, e.g. `--ktop 5`

The pre-trained models both for synthetic and real datasets are already in `data` folder. one may train their ow model by specifying `--train_model 1`


### III. MNIST dataset

#### 1. Train posthoc accuracy model: 


```code/mnist_posthoc_accuracy_eval.py```

#### 2. Run BIF: 

```code/switch_mnist_featimp.py``` 

## Benchmark methods


### Table 1: Synthetic datasets (also used for fig 2.)
 
L2X: `comparison_methods/L2X/explain_invase_data.py` \
INVASE: `comparison_methods/INVASE/invase_synth_runs.sh` (calls `main_invase.py`)


### Table 3: MNIST data

L2X: `comparison_methods/L2X/l2x_mnist_patch_exp.py` \
INVASE: `comparison_methods/INVASE/invase_mnist_patch_exp.py` 

### Figure 3: Fairness trade-off

Classifier: `code/Trade_offs/vfairness_weight_readout.py` \
INVASE: `comparison_methods/INVASE/invase_fair_adult.py`

### Figure 4: Privacy trade-off

Classifier: `code/Trade_offs/fairness_vs_privacy.py` \
INVASE: `comparison_methods/INVASE/invase_private_adult.py`






