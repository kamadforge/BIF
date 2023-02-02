from train_network import get_args, get_data, get_net, train, test

args = get_args()
X, Xtst, y, ytst, x_tot, y_tot, datatypes_tr, datatypes_tst, datatypes_tst_num_relevantfeatures = get_data(args)
model, criterion, optimizer = get_net(x_tot, args)
num_epochs = args.train_epochs

if args.mode == 'train':
    train(args, model, optimizer,criterion, X, Xtst, y, ytst)
###########################################
# TEST FOR SYNTHETIC DATASETS (GLOBAL AND LOCAL) (TABLE 1)
elif args.mode == "test":
    test(args)