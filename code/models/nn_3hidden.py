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
    def __init__(self, input_num, output_num, hidden_dim):
        super(FC_net, self).__init__()

        # change 200 to hidden_dim
        self.fc1 = nn.Linear(input_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(hidden_dim, output_num)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.bn1(output)
        output = nn.functional.relu(self.fc2(output))
        output = self.bn2(output)
        # output = self.fc3(output)
        output = self.fc4(output)
        return output

