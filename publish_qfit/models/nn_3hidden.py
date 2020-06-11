import torch.nn as nn



class FC(nn.Module):
    def __init__(self, input_num, output_num):
        super(FC, self).__init__()

        # change 200 to hidden_dim
        self.fc1=nn.Linear(input_num,200)
        self.fc2=nn.Linear(200,200)
        #self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, output_num)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

    def forward(self, x):

        output=self.fc1(x)
        output=self.bn1(output)
        output=nn.functional.relu(self.fc2(output))
        output = self.bn2(output)
        #output = self.fc3(output)
        output = self.fc4(output)
        return output


class FC_net(nn.Module):
    def __init__(self, input_num, output_num, hidden_dim, which_norm):
        super(FC_net, self).__init__()

        # change 200 to hidden_dim
        self.fc1 = nn.Linear(input_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_num)
        self.relu = nn.ReLU()

        if which_norm=='batch_norm':
            self.which_norm = 'batch_norm'
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        elif which_norm=='weight_norm':
            self.which_norm = 'weight_norm'
            self.fc1 = nn.utils.weight_norm(nn.Linear(input_num, hidden_dim), name='weight')
            self.fc2 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim), name='weight')
        elif which_norm=='spectral_norm':
            self.which_norm = 'spectral_norm'
            self.fc1 = nn.utils.spectral_norm(nn.Linear(input_num, hidden_dim), name='weight')
            self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim), name='weight')


    def forward(self, x):

        if self.which_norm=='batch_norm':

            output = self.fc1(x)
            output = self.bn1(output)
            output = self.relu(self.fc2(output))
            output = self.bn2(output)
            output = self.fc3(output)

        else:

            hidden = self.fc1(x)
            output = self.fc2(self.relu(hidden))
            output = self.relu(output)
            output = self.fc3(output)


        return output

