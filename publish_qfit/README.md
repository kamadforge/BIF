# Q-FIT


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
 


###I. Synthetic tabular datasets:

#### 1. Get the pre-trained models 
(optional: in case no other desired model exists)

```
cd code
python train_network.py --dataset xor
```

Datasets: xor, orange_skin, nonlinear_additive, syn4, syn5, syn6

The pre-trained models are already in `code/checkpoints directory`

#### 2. Run Q-FIT and compute metrics

To compute TPR/FDR (Experiment 1) and see the average feature ranking:

```
python featimp_test.py --dataset xor
```


###II. Real-world tabular datasets

####1. Pre-train the network

```
cd code
python train_network.py --dataset adult_short
```

Datasets: credit, adult_short

The pre-trained models are already in `data` folder.

#### 2. Run Q-FIT and compute feature ranking

To compute the average feature ranking:

```
python featimp_test.py --dataset adult_short
```

#### 3. Test how important features perform

And to test how the dataset with reduced (k) number of features performs (Experiment 2)

```
python train_network.py --dataset adult_short --mode test --prune True --k 3
```

### III. MNIST dataset

#### 1. Train posthoc accuracy model: 


```code/mnist_posthoc_accuracy_eval.py```

#### 2. Run Q-fit: 

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






