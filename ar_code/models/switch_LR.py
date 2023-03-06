import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.distributions import Gamma


class Model(nn.Module):

    def __init__(self, input_dim, LR_model, num_samps_for_switch):

        super(Model, self).__init__()

        self.W = LR_model
        self.parameter = Parameter(-1e-10*torch.ones(input_dim),requires_grad=True)
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

        x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples
        x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W))) #[100,150]
        labelstack = torch.sigmoid(x_out)

        return labelstack, phi