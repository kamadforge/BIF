from train_network import get_data, get_net, test
from train_network import train as train_network
from featimp import shuffle_data, loss_function
from featimp import test_pruned_syn, train_switches, test_get_switches, test_pruned


from models.switch_MLP import Model_switchlearning #switch_nn/local
from models.switch_MLP import Modelnn #global
import argparse
import os
import numpy as np
import socket
import sys
from pathlib import Path
import torch

########################3
# ARGS

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--dataset", default="xor") #xor, orange_skin, nonlinear_additive, alternating, syn4, syn5, syn6, adult_short, credit, intrusion
    parser.add_argument("--load_dataset", default=1, type=int)
    parser.add_argument("--method", default="nn")
    parser.add_argument("--mini_batch_size", default=100, type=int)
    parser.add_argument("--epochs", default=5, type=int) # 7
    parser.add_argument("--lr", default=0.01, type=float)
    # for switch training
    parser.add_argument("--num_Dir_samples", default=200, type=int)
    parser.add_argument("--alpha", default=0.0001, type=float)
    parser.add_argument("--kl_term", default=1, type=int)

    parser.add_argument("--point_estimate", default=0, type=int)
    parser.add_argument("--switch_nn", default=1, type=int)

    parser.add_argument("--train_model", default=0, type=int)
    parser.add_argument("--train_switches", default=1, type=int)
    parser.add_argument("--test_switches", default=1, type=int)
    # for instance wise training switch_nn=1, and 0 for global

    parser.add_argument("--training_local", default=0, type=int)
    parser.add_argument("--local_training_iter", default=200, type=int)
    parser.add_argument("--set_hooks", default=1, type=int)


    parser.add_argument("--ktop_real", default=3, type=int)
    parser.add_argument("--runs_num", default=5, type=int)

    # training a mdel
    parser.add_argument("--mode", default="train") # test, train
    parser.add_argument("--testtype", default="local") #global, local
    parser.add_argument("--prune", default=True) #tests the subset of features
    parser.add_argument("--ktop", default=5, type=int)
    parser.add_argument("--met", default=4, type=int) #0-bif,  1-shap, 2-invase 3-l2x 4-lime
    parser.add_argument("--train_epochs", default=500, type=int) #500
    parser.add_argument("--which_net", default="FC")
    # parse
    args = parser.parse_args()
    return args

########################################
# PATH

cwd = os.getcwd()
cwd_parent = Path(__file__).parent.parent
if socket.gethostname()=='worona.local':
    pathmain = cwd
    path_code = os.path.join(pathmain, "code")
elif 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    sys.path.append(os.path.join(cwd_parent, "data"))
    pathmain=cwd
    path_code = os.path.join(pathmain, "code")
    #path_code = os.path.join(pathmain)
else:
    pathmain = cwd_parent
    path_code = cwd

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints_bif", exist_ok=True)

#######################################
# DATA

synthetic = ["xor", "subtract", "xor_mean5", "orange_skin", "orange_skin_mean5", "nonlinear_additive", "alternating", "syn4", "syn4_mean5", "syn5", "syn6"]

args = get_args()
X, Xtst, y, ytst, x_tot, y_tot, datatypes_tr, datatypes_tst, datatypes_tst_num_relevantfeatures = get_data(args, synthetic)

#####################################
# TRAINING MODEL with data from scratch. this model is used to infer the feature importance weights

model, criterion, optimizer = get_net(x_tot, args)
num_epochs = args.train_epochs

checkpoints = {
    "subtract": "checkpoints/subtract_nn_LR_model0_epochs_500_acc_0.99",
    "xor": "checkpoints/xor_nn_LR_model0_epochs_500_acc_0.97"
}
if args.train_model and args.mode == 'train':
    saved_model_path = train_network(args, model, optimizer,criterion, X, Xtst, y, ytst)
else:
    saved_model_path = checkpoints[args.dataset]
##############
# TEST FOR SYNTHETIC DATASETS (GLOBAL AND LOCAL) (TABLE 1)
# elif args.mode == "test":
#     test(args)


if "syn" in args.dataset and not args.switch_nn:
    print("\n\n Please use local setting to test alternating datasets")
    exit()

#####################################
# TRAIN SWITCHES (BOTH LOCAL AND GLOBAL)

# load pretrained model g for the switch model

loaded_model = np.load(saved_model_path+".npy", allow_pickle=True)
print("Loaded: ", saved_model_path)
# if 1:
#     for i in os.listdir("checkpoints"):
#         file = os.path.join(path_code, "checkpoints", i)
#         if args.dataset in file and args.method in file:
#             loaded_model = np.load(file, allow_pickle=True)
#             model_g = file
#             print("Loaded: ", file)
#     if not os.path.isdir("weights"):
#         os.mkdir("weights")

if args.train_switches:
    train_switches(args, loaded_model, X, Xtst, y, ytst, datatypes_tr, datatypes_tst)

####################################3
# TEST SWITCHES (BOTH LOCAL AND GLOBAL)

if args.test_switches:

    print("\nTesting:\n")

    global output_num
    if args.dataset == "intrusion":
        output_num = 4
    else:
        output_num = 2


    if args.switch_nn:
        # getting lcaol switches from the importance net
        S, datatypes_test_samp_arg, datatypes_test_samp_onehot, inputs_test_samp = test_get_switches(args.dataset, args.switch_nn, False, output_num, Xtst, ytst, datatypes_tst, args)
        print("Got local switches from the importance network")
        if not args.point_estimate:
            S = torch.mean(S, axis=2)
    else:  # get global switches
        S = torch.load(f"rankings/global/global_{args.dataset}_pointest_{args.point_estimate}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt")
        print(f"Switch global loaded: {S}")
        #S = S.unsqueeze(0).repeat(X_test.shape[0],1)
        inputs_test_samp = torch.Tensor(Xtst)
        datatypes_test_samp_arg = None


        # testing the local switches


    if (args.dataset in synthetic):
        accuracy = test_pruned_syn(S, args, Xtst, ytst, datatypes_tst, datatypes_test_samp_arg, synthetic)
    else:
        accuracy = test_pruned(S, inputs_test_samp, args.ktop_real, model_g, Xtst, ytst, args)

    print("Tested on the subset of features chosen for each instance")


    #return [accuracy]