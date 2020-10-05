# Q-FIT


## requirements

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
 
### Table 1: Synthetic datasets (also used for fig 2.)
 
L2X: `comparison_methods/L2X/explain_invase_data.py` \
INVASE: `comparison_methods/INVASE/invase_synth_runs.sh` (calls `main_invase.py`)


### Table 3: MNIST data

Train posthoc accuracy model: `code/mnist_posthoc_accuracy_eval.py` \
Q-fit: `code/switch_mnist_featimp.py` \
L2X: `comparison_methods/L2X/l2x_mnist_patch_exp.py` \
INVASE: `comparison_methods/INVASE/invase_mnist_patch_exp.py` 

### Figure 3: Fairness trade-off

Classifier: `code/Trade_offs/vfairness_weight_readout.py` \
INVASE: `comparison_methods/INVASE/invase_fair_adult.py`

### Figure 4: Privacy trade-off

Classifier: `code/Trade_offs/fairness_vs_privacy.py` \
INVASE: `comparison_methods/INVASE/invase_private_adult.py`

