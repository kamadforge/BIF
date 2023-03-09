import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.distributions import Gamma
import numpy as np




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

        pre_phi=self.parameter
        phi = F.softplus(self.parameter)
        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        # compute the variance

        var_phi = ((phi/torch.sum(phi))*(1-(phi/torch.sum(phi))))/(torch.sum(phi)+1)

        # this is the mean which we use as a proxy for S importance
        if self.point_estimate:
            S = phi / torch.sum(phi)
            output = x * S
        else:

            """ draw Gamma RVs using phi and 1 """
            num_samps = self.num_samps_for_switch
            concentration_param = phi.view(-1,1).repeat(1,num_samps) #[feat x sampnum]
            beta_param = torch.ones(concentration_param.size()) #[feat x sampnum]
            #Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
            Gamma_obj = Gamma(concentration_param, beta_param) #[feat x sampnum]
            gamma_samps = Gamma_obj.rsample() #200, 150, input_dim x samples_num

            if any(torch.sum(gamma_samps,0)==0):
                print("sum of gamma samps are zero!")
            else:
                Sstack = gamma_samps / torch.sum(gamma_samps, 0) # input dim by  # samples

            #x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples

            SstackT = Sstack.t() # Dirichlet samples by mini_batch
            output, Sprime = self.switch_func_fc(x, SstackT)
            S=SstackT
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
            output = output.reshape(self.num_samps_for_switch, x.shape[0], -1) #changed mini_batch_size to x.shape[0]
            output = output.transpose_(0, 1)


        return output, phi, S, pre_phi, var_phi

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


# local setting, network to learn switch through the forward pass
class Model_switchlearning(nn.Module):

    def __init__(self, input_num, output_num, num_samps_for_switch, mini_batch_size, point_estimate):
    # def __init__(self, input_dim, hidden_dim):
        super(Model_switchlearning, self).__init__()
        # local importance net
        impnet_num =200
        self.phi_fc1 = nn.Linear(input_num, impnet_num)
        self.phi_fc2 = nn.Linear(impnet_num,impnet_num)
        #self.phi_fc2b = nn.Linear(200, 200)
        self.phi_fc3 = nn.Linear(impnet_num, input_num) #outputs switch values
        self.fc1_bn1 = nn.BatchNorm1d(impnet_num)
        #self.fc2_bn2b = nn.BatchNorm1d(200)
        self.fc2_bn2 = nn.BatchNorm1d(impnet_num)
        #self.parameters = -1e-10*torch.ones(input_num) #just an output
        self.num_samps_for_switch = num_samps_for_switch
        self.mini_batch_size = mini_batch_size
        # model g
        self.fc1 = nn.Linear(input_num, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, output_num)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

        self.point_estimate = point_estimate

    # def switch_func_fc(self, output, SstackT):
    #
    #     # output is (100,10,24,24) or (110,10), we want to have 100 (batch),150 (samp),10,24,24, I guess
    #     #output = torch.einsum('ij, mj -> imj', (SstackT, output)) # samples, batchsize, dimension
    #     output = torch.einsum('imj, mj -> imj', (SstackT, output)) # samples, batchsize, dimension
    #
    #     output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
    #
    #     return output, SstackT

    def forward(self, x, mini_batch_size): # x is mini_batch_size by input_dim

        output = self.phi_fc1(x)
        #output = nn.functional.relu(self.phi_fc1(x))
        output = self.fc1_bn1(output)
        output = nn.functional.relu(self.phi_fc2(output))
        output = self.fc2_bn2(output)
        #output = nn.functional.relu(self.phi_fc2b(output))
        #output = self.fc2_bn2b(output)
        pre_phi = self.phi_fc3(output)
        # phi = F.softplus(phi_parameter.mean(dim=0))
        phi = F.softplus(pre_phi) # now the size of phi is mini_batch by input_dim
        # [200,11]

        if self.point_estimate:
            # S = phi / torch.sum(phi)
            # there is a switch vector for each sample
            S = phi/torch.sum(phi,dim=1).unsqueeze(dim=1) #[batch x featnum]
            output = x * S
            output = self.fc1(output)  # samples*batchsize, dimension
            output = self.bn1(output)
            output = nn.functional.relu(self.fc2(output))
            output = self.bn2(output)
            output = self.fc4(output)

        else:
            """ draw Gamma RVs using phi and 1 """
            #previously we had one switch vector we learnt
            #now we are learning switch for every sample
            #previously we draw [feat_dim x samples_num]
            #now we need to account for a sample
            num_samps = self.num_samps_for_switch
            feat_dim = phi.shape[1]
            # sanity check: we draw samples for each data sample in a for loop first and see if this statistic matches full one
            compute_loop = False
            if compute_loop:
                S = torch.zeros((self.mini_batch_size, feat_dim, num_samps))
                for i in torch.arange(0, self.mini_batch_size):
                    this_phi_vec = phi[i,:]
                    """ draw Gamma RVs using phi and 1 """
                    concentration_param = this_phi_vec.view(-1, 1).repeat(1, num_samps)  # [feat x sampnum]
                    beta_param = torch.ones(concentration_param.size())  # [feat x sampnum]
                    # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
                    Gamma_obj = Gamma(concentration_param, beta_param)  # [feat x sampnum]
                    gamma_samps = Gamma_obj.rsample()  # feat x samples_num

                    if any(torch.sum(gamma_samps, 0) == 0):
                        print("sum of gamma samps are zero!")
                    else:
                        Sstack = gamma_samps / torch.sum(gamma_samps, 0)  # input dim by  # samples

                    S[i,:,:] = Sstack
            else:
                concentration_param = phi[:, :, None].expand(-1, -1, num_samps)
                beta_param = torch.ones_like(concentration_param)
                Gamma_obj = Gamma(concentration_param, beta_param)
                gamma_samps = Gamma_obj.rsample()  # bs x feat_dim x samples_num
                norm_sum = torch.sum(gamma_samps, 1, keepdim=True)
                assert not (norm_sum == 0).any()
                S = gamma_samps / norm_sum  # normalize in feature dimension

            x_samps = torch.einsum("ij,ijk -> ikj", (x, S)) # x: minibatch by feat_dim, samps_mat: minibatch by feat_dim by num_samps
            bs, n_samp, n_feat = x_samps.shape
            output = x_samps.reshape(bs * n_samp, n_feat)
            # model_out = self.trained_model(x_samps)
            # model_out = model_out.view(bs, n_samp)

            output = self.fc1(output)  # samples*batchsize, dimension
            output = self.bn1(output)
            output = nn.functional.relu(self.fc2(output))
            output = self.bn2(output)
            output = self.fc4(output)
            output_dim = output.shape[1]
            output = output.view(bs, n_samp, output_dim)

            # concentration_param = phi.view(-1, 1).repeat(1, num_samps)
            # beta_param = torch.ones(concentration_param.size())
            # # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
            # Gamma_obj = Gamma(concentration_param, beta_param)
            # gamma_samps = Gamma_obj.rsample()  # 200, 150, feat_dim x samples_num


            # if any(torch.sum(gamma_samps, 0) == 0):
            #     print("sum of gamma samps are zero!")
            # else:
            #     Sstack = gamma_samps / torch.sum(gamma_samps, 0)  # input dim by  # samples
            #     #S = Sstack
            # # x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples
            #
            # #Sprime = Sstack.t()  # Dirichlet samples by mini_batch
            # Sprime = Sstack.reshape(num_samps, phi.shape[0], -1)
            # output, Sprime = self.switch_func_fc(x, Sprime)
            #
            # S = Sprime.mean(dim=0)

        # if not self.point_estimate:
        #     output = output.reshape(self.num_samps_for_switch, mini_batch_size, -1)
        #     output = output.transpose_(0,1)

        # S: [batch x feat]
        return output, phi, S, pre_phi, None



class ThreeNet(nn.Module):

  def __init__(self, baseline_net, classifier_net, switch_net, input_num, output_num, num_samps_for_switch, mini_batch_size, point_estimate):
    # def __init__(self, input_dim, hidden_dim):
    super(ThreeNet, self).__init__()

    self.trained_model = baseline_net
    self.switch_net = switch_net
    self.classifier_net = classifier_net

    self.num_samps_for_switch = num_samps_for_switch
    self.mini_batch_size = mini_batch_size
    self.point_estimate = point_estimate

  # def switch_func_fc(self, output, SstackT):
  #
  #     # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
  #     output = torch.einsum('ij, mj -> imj', (SstackT, output))  # samples, batchsize, dimension
  #     output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
  #
  #     return output, SstackT

  def forward(self, x, mini_batch_size):  # x is mini_batch_size by input_dim

      baseline_net_output = self.trained_model(x)

      pre_phi = self.switch_net(x)
      phi = F.softplus(pre_phi)  # now the size of phi is mini_batch by input_dim

      if torch.sum(torch.isnan(phi))>=1:
          print("some Phis are NaN")

      if self.point_estimate:
          S = phi / torch.sum(phi, dim=1).unsqueeze(dim=1)
          output = x * S

      # norm_preserving = True
      # if norm_preserving:
      #     x_norm = torch.norm(x,dim=1).unsqueeze(dim=1)
      #     output_norm = torch.norm(output,dim=1).unsqueeze(dim=1)
      #     output = output/output_norm*x_norm

      output = self.classifier_net(output)

      # if not self.point_estimate:
      #     output = output.reshape(self.num_samps_for_switch, mini_batch_size, -1)
      #     output = output.transpose_(0, 1)

      return output, phi, S, pre_phi, baseline_net_output
