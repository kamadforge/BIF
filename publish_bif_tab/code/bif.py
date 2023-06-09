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
import yaml
from datetime import datetime
########################3
# ARGS

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--dataset", default="xor") #xor, orange_skin, nonlinear_additive, alternating, syn4, syn5, syn6, adult_short, credit, intrusion
    parser.add_argument("--load_dataset", default=1, type=int)
    parser.add_argument("--method", default="nn")
    parser.add_argument("--batch", default=200, type=int)
    parser.add_argument("--epochs", default=5, type=int) # 7
    parser.add_argument("--lr", default=0.01, type=float)
    # for switch training
    parser.add_argument("--num_Dir_samples", default=200, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--kl_term", default=1, type=int)

    parser.add_argument("--point_estimate", default=1, type=int)
    parser.add_argument("--switch_nn", default=0, type=int)

    parser.add_argument("--train_model", default=0, type=int)
    parser.add_argument("--train_switches", default=1, type=int)
    parser.add_argument("--test_switches", default=1, type=int)
    # for instance wise training switch_nn=1, and 0 for global

    parser.add_argument("--training_local", default=0, type=int)
    parser.add_argument("--local_training_iter", default=200, type=int)
    parser.add_argument("--set_hooks", default=1, type=int)
    parser.add_argument("--load_params", default=1, type=int)

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

    if args.load_params:
        opt = yaml.load(open("utils/params.yml"), Loader=yaml.FullLoader)
        for key, params in opt.items():
            if key==args.dataset:
                for key, value in params.items():
                    setattr(args, key, value)
        print("arguments: {}".format(str(args)))

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
    "subtract": "checkpoints/subtract_nn_LR_model0_epochs_500_acc_0.99.npy",
    "xor": "checkpoints/xor_nn_LR_model0_epochs_500_acc_0.97.npy",
    "orange_skin": "checkpoints/orange_skin_nn_LR_model0_epochs_500_acc_1.00.npy",
    "nonlinear_additive": "checkpoints/nonlinear_additive_nn_LR_model0_epochs_500_acc_0.98.npy",
    "syn4": "checkpoints/syn4_nn_LR_model0_epochs_500_acc_0.65.npy",
    "syn5": "checkpoints/syn5_nn_LR_model0_epochs_500_acc_0.68.npy",
    "syn6": "checkpoints/syn6_nn_LR_model0_epochs_500_acc_0.74.npy",
    "adult_short": "checkpoints/adult_short_nn_LR_model0_epochs_2000_acc_0.84.npy",
    "credit": "checkpoints/credit_nn_LR_model0_epochs_2000_acc_0.97.npy",
    "intrusion": "checkpoints/intrusion_nn_LR_model0_epochs_250_acc_0.96.npy"

}

if not args.train_model and args.dataset not in checkpoints:
    print(f"No checkpoint for {args.dataset}. Please add path to checkpoints or select args.train_model")
    exit()

if args.train_model and args.mode == 'train':
    saved_model_path = train_network(args, model, optimizer,criterion, X, Xtst, y, ytst) + ".npy"
else:
    saved_model_path = checkpoints[args.dataset]
##############
# TEST FOR SYNTHETIC DATASETS (GLOBAL AND LOCAL) (TABLE 1)


if "syn" in args.dataset and not args.switch_nn:
    print("\n\n Please use local setting to test alternating datasets")
    exit()

#####################################
# TRAIN SWITCHES (BOTH LOCAL AND GLOBAL)

# load pretrained model g for the switch model
loaded_model = np.load(saved_model_path, allow_pickle=True)
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
    switch_path = train_switches(args, loaded_model, X, Xtst, y, ytst, datatypes_tr, datatypes_tst)

####################################3
# TEST SWITCHES (BOTH LOCAL AND GLOBAL)

if args.test_switches:

    print("\nTesting:\n")

    global output_num
    if args.dataset == "intrusion":
        output_num = 4
    else:
        output_num = 2

    # get switches
    if args.switch_nn:  # getting lcaol switches from the importance net
        S, datatypes_test_samp_arg, datatypes_test_samp_onehot, inputs_test_samp = test_get_switches(args.dataset, args.switch_nn, False, output_num, Xtst, ytst, datatypes_tst, args, switch_path)
        print("Got local switches from the importance network")
        if not args.point_estimate:
            S = torch.mean(S, axis=2)
    else:  # get global switches
        if switch_path is None:
            global_switch_path = f"rankings/global/global_{args.dataset}_pointest_{args.point_estimate}_batch_{args.batch}_lr_{args.lr}_epochs_{args.epochs}_alpha_{args.alpha}.pt"
        else:
            global_switch_path = switch_path
        ts = os.path.getmtime(global_switch_path)
        print("switches from: ", global_switch_path,"\n", datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
        S = torch.load(global_switch_path)
        print(f"Switch global loaded: {S}")
        #S = S.unsqueeze(0).repeat(X_test.shape[0],1)
        inputs_test_samp = torch.Tensor(Xtst)
        datatypes_test_samp_arg = None


        # testing the local switches


    if (args.dataset in synthetic):
        # mcc how many important features have been detected (true positive features)
        accuracy = test_pruned_syn(S, args, Xtst, ytst, datatypes_tst, datatypes_test_samp_arg, synthetic)
    else:
        # classification accuracy with a subset of features
        accuracy = test_pruned(S, inputs_test_samp, args.ktop_real, saved_model_path, Xtst, ytst, args)

    print("Tested on the subset of features chosen for each instance")


    #return [accuracy]

    datetime_uni = datetime.now()
    date = datetime_uni.strftime("%d/%m/%y")
    date_time = datetime_uni.strftime('%Y-%m-%d %H:%M:%S')
    print(date_time)
    def write_to_csv(args, accuracy, date):
        # file write
        os.makedirs("results", exist_ok=1)
        filename = f"results/grad_results_{args.dataset}.csv"
        file = open(filename, "a+")
        file.write(
            f"{args.point_estimate}, {args.switch_nn}, {args.kl_term}, {args.alpha}, {args.batch}, {args.epochs}, {args.lr}, {args.num_Dir_samples},-,-,{accuracy},-,{date}\n")
        file.close()

    write_to_csv(args, accuracy, date)
