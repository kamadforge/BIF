import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.bn1(hidden)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.bn2(output)
        output = self.fc3(self.relu(output))
        output = self.sigmoid(output)
        return output


class Feature_Importance_Model(nn.Module):
    #I'm going to define my own Model here following how I generated this dataset
    def __init__(self, input_dim, classifier, num_samps_for_switch):
    # def __init__(self, input_dim, hidden_dim):
        super(Feature_Importance_Model, self).__init__()
        self.classifier = classifier
        self.parameter = Parameter(-1e-10*torch.ones(input_dim),requires_grad=True) # this parameter lies
        self.num_samps_for_switch = num_samps_for_switch
    def forward(self, x): # x is mini_batch_size by input_dim
        phi = F.softplus(self.parameter)
        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)
        """ draw Gamma RVs using phi and 1 """
        num_samps = self.num_samps_for_switch
        concentration_param = phi.view(-1,1).repeat(1,num_samps)
        beta_param = torch.ones(concentration_param.size())
        #Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample() #200, 150, input_dim x samples_num
        if any(torch.sum(gamma_samps,0)==0):
            print("sum of gamma samps are zero!")
        else:
            Sstack = gamma_samps / torch.sum(gamma_samps, 0) # input dim by  # samples
        x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) # batch by input dim by Dir samps

        # reshaping x_samps such that the outcome is (batch*Dir samps) by input dim
        x_samps = x_samps.transpose_(1, 2) # batch by Dir samps by input dim
        batch, Dir_samps, input_dim  = x_samps.shape
        x_samps = x_samps.reshape(batch * Dir_samps, input_dim)

        x_out = self.classifier(x_samps)
        labelstack = torch.sigmoid(x_out)

        # reshaping back to batch by Dir samps
        labelstack = labelstack.reshape(batch, Dir_samps)
        return labelstack, phi