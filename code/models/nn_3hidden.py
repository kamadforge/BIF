import torch.nn as nn



class FC(nn.Module):
    def __init__(self, input_num, output_num):
        super(FC, self).__init__()

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

