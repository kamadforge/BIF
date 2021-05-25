from sklearn.model_selection import ParameterGrid
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="adult_short")
parser.add_argument("--ktop_real", default=1, type=int)
args = parser.parse_args()

params=ParameterGrid({"dataset": [args.dataset], "lr": [0.005, 0.1, 0.05, 0.01], "epochs":[5,6,7,8,9,10,11,12,13, 14, 15,20], "mini_batch_size": [10,50,110,200], "ktop_real": [args.ktop_real]})

for p in params:

    print(p)

    subprocess.call(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/featimp_dp/code/featimp_test.py", "--dataset", p['dataset'], "--lr", str(p['lr']), "--epochs", str(p['epochs']), "--mini_batch_size", str(p['mini_batch_size']), "--ktop_real", str(p['ktop_real'])])

    #subprocess.call(["python", "featimp_test.py", "--dataset", p['dataset'], "--lr", str(p['lr']), "--epochs", str(p['epochs']), "--mini_batch_size", str(p['mini_batch_size']), "--ktop_real", str(p['ktop_real'])])