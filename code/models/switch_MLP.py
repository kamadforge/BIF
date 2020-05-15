import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.distributions import Gamma


class Modelnn(nn.Module):

    def __init__(self, input_num, output_num, num_samps_for_switch, mini_batch_size, point_estimate):
    # def __init__(self, input_dim, hidden_dim):
        super(Modelnn, self).__init__()

        #self.W = LR_model
        self.parameter = Parameter(-1e-10*torch.ones(input_num),requires_grad=True)
        self.num_samps_for_switch = num_samps_for_switch
        self.mini_batch_size = mini_batch_size

        self.fc1 = nn.Linear(input_num, 200)
        self.fc2 = nn.Linear(200, 200)
        # self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, output_num)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

        self.point_estimate = point_estimate

    def switch_func_fc(self, output, SstackT):

        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output = torch.einsum('ij, mj -> imj', (SstackT, output)) # samples, batchsize, dimension
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        return output, SstackT

    def forward(self, x, mini_batch_size): # x is mini_batch_size by input_dim

        if x.shape[0]==1:
            dummy=0

        phi = F.softplus(self.parameter)

        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        if self.point_estimate:
            S = phi / torch.sum(phi)
            output = x * S
        else:

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

            #x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples

            SstackT = Sstack.t() # Dirichlet samples by mini_batch
            output, Sprime = self.switch_func_fc(x, SstackT)

            # x_samps = torch.einsum("ij,jk -> ijk", (x, Sstack))  # ([150, 10, 100]) batch size, dim, samples
            # x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W)))  # [100,150]
            # labelstack = torch.sigmoid(x_out)


        output = self.fc1(output) # samples*batchsize, dimension
        output = self.bn1(output)
        output = nn.functional.relu(self.fc2(output))
        output = self.bn2(output)
        # output = self.fc3(output)
        output = self.fc4(output)

        if not self.point_estimate:
            output = output.reshape(self.num_samps_for_switch, mini_batch_size, -1)
            output = output.transpose_(0, 1)

        return output, phi

    # t = torch.tensor([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]], [[17,18,19,20],[21,22,23,24]]])
    # t.shape is torch.Size([3, 2, 4])
    # tensor([[[ 1,  2,  3,  4],
    #          [ 5,  6,  7,  8]],
    #         [[ 9, 10, 11, 12],
    #          [13, 14, 15, 16]],
    #         [[17, 18, 19, 20],
    #          [21, 22, 23, 24]]])
    #
    # t.shape is torch.Size([3, 2, 4])
    # tt = t.reshape(3*2,4)
    # tt.reshape(2,3,-1) is
    # tensor([[[ 1,  2,  3,  4],
    #          [ 5,  6,  7,  8],
    #          [ 9, 10, 11, 12]],
    #         [[13, 14, 15, 16],
    #          [17, 18, 19, 20],
    #          [21, 22, 23, 24]]])
    # hence, the final reshape has to be
    # tt.reshape(3,2,-1)
    #
    # tt.reshape(3, 2, -1)
    # tensor([[[1, 2, 3, 4],
    #          [5, 6, 7, 8]],
    #         [[9, 10, 11, 12],
    #          [13, 14, 15, 16]],
    #         [[17, 18, 19, 20],
    #          [21, 22, 23, 24]]])


# network to learn switch through the forward pass

class Model_switchlearning(nn.Module):

    def __init__(self, input_num, output_num, num_samps_for_switch, mini_batch_size, point_estimate):
    # def __init__(self, input_dim, hidden_dim):
        super(Model_switchlearning, self).__init__()


        self.phi_fc1 = nn.Linear(input_num, 100)
        self.phi_fc2 = nn.Linear(100,100)
        self.phi_fc3 = nn.Linear(100, input_num) #outputs switch values

        self.fc1_bn1 = nn.BatchNorm1d(100)
        self.fc2_bn2 = nn.BatchNorm1d(100)

        #self.parameters = -1e-10*torch.ones(input_num) #just an output

        self.num_samps_for_switch = num_samps_for_switch
        self.mini_batch_size = mini_batch_size

        self.fc1 = nn.Linear(input_num, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, output_num)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

        self.point_estimate = point_estimate

    def switch_func_fc(self, output, SstackT):

        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output = torch.einsum('ij, mj -> imj', (SstackT, output)) # samples, batchsize, dimension
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        return output, SstackT

    def forward(self, x, mini_batch_size): # x is mini_batch_size by input_dim

        if x.shape[0]==1:
            dummy=0

        output = self.phi_fc1(x)
        output = self.fc1_bn1(output)
        output = nn.functional.relu(self.phi_fc2(output))
        output = self.fc2_bn2(output)
        phi_parameter = self.phi_fc3(output)

        phi = F.softplus(phi_parameter.mean(dim=0))

        if self.point_estimate:
            S = phi / torch.sum(phi)
            output = x * S

        output = self.fc1(output) # samples*batchsize, dimension
        output = self.bn1(output)
        output = nn.functional.relu(self.fc2(output))
        output = self.bn2(output)
        output = self.fc4(output)

        if not self.point_estimate:
            output = output.reshape(self.num_samps_for_switch, mini_batch_size, -1)
            output = output.transpose_(0,1)

        return output, phi

