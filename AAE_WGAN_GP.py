# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:37:53 2021
@author: Hristo Petkov
"""

"""
@inproceedings{yu2019dag,
  title={DAG-GNN: DAG Structure Learning with Graph Neural Networks},
  author={Yue Yu, Jie Chen, Tian Gao, and Mo Yu},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
"""

import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.linalg as slin
import os

from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import Linear, Sequential, LeakyReLU, Dropout, BatchNorm1d
from Utils import preprocess_adj_new, preprocess_adj_new1
from Utils import nll_gaussian, kl_gaussian_sem,  nll_catogrical
from Utils import _h_A
from Utils import count_accuracy  
    
class Discriminator(nn.Module):
    """Discriminator module."""
    def __init__(self, input_dim, discriminator_dim, negative_slope, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        self.negative_slope = negative_slope
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(self.negative_slope), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)
        self.init_weights()
        
        self.var_out_ftrs = [3, 4, 1, 1, 1, 1, 1, 1, 1, 1] 
        # discrete var 1:
        self.embed1 =   nn.Embedding(self.var_out_ftrs[0], self.var_out_ftrs[0])
        self.linear_map1 = nn.Linear(self.var_out_ftrs[0], 1)
        # discrete var 2:
        self.embed2 =   nn.Embedding(self.var_out_ftrs[1], self.var_out_ftrs[1])
        self.linear_map2 = nn.Linear(self.var_out_ftrs[1], 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def calc_gradient_penalty(self, real_data, fake_data, data_type, device='cpu', pac=10, lambda_=10):
        
        # reshape data
        real_data = real_data.squeeze()
        fake_data = fake_data.squeeze()
        
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty
        
    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        # batch size must be divisible by pac=10
        discrete_vars, cont_vars = input['discrete'], input['continuous']
        
        ''' map discrete variables through embedding then linear layer '''
        d1 = discrete_vars[0]
        d1 = self.linear1(self.embed1(d1)) 
        # variable is now 1-dim real --> concatenate

        d2 = discrete_vars[1]
        d2 = self.linear2(self.embed2(d2)) 
        # variable is also now 1-dim real --> concatenate

        # concatenate all variables (each 1-dim)
        x = torch.cat((d1, d2, cont_vars), dim=1)

        return self.seq(x.view(-1, self.pacdim))
class Generator(nn.Module):
    """Generator module (based on DAG-NotearsMLP)
        
    Changes by Calum (5th Oct 2022)
    - Generator:
        - new attributes: lines 129--146.
            --> var_out_ftrs: for specifying out_features needed in each local linear layer
            --> embed and linear_map: for mapping discrete vars into contin. representations
            --> (l 154--161) replaced for loop to iterate over each dth variable separately,
                rather than constructing all 10 local layers at same time.
                means we can specify output dim for each var (e.g 3 for discrete, 1 for cont.). 
            --> added "if var_dim>1: then discrete=True" flag for F.softmax() functionality.

        - forward(): lines 179--190.
            --> added processing steps to make sure we can concatenate 
                discrete + continuous into one vector X.
                makes use of embedding+mapping (see last point).
            --> (194--200) concatenate noise with hidden act for sampling each local layer.
            --> added flag for passing x_tilde through F.softmax() if discrete
           
        - LocallyConnected: line 241 etal.
            --> changed attributes, so now num_linear is removed. 
                reason is that we need to construct the weight(+bias) matrix dimension
                of each local linear separately now.
            --> input_features = d+1: we are looking to concatenate noise (z)
                into each local layer, which will have channel_dim=1.
            --> added discrete flag to control whether F.softmax() used or not 

    """
    def __init__(self, dims, args, bias=True):
        super(Generator, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims

        '''____________________ new attributes 
        '''
        # out_features for each variable: here, first 2 vars are discrete, rest are contin.
        # with 3 and 4 classes, respectively (e.g. d1={low, med, high}, d2={cat,dog,sheep,cow})
        self.var_out_ftrs = [3, 4, 1, 1, 1, 1, 1, 1, 1, 1] 
        # discrete var 1:
        self.embed1 =   nn.Embedding(self.var_out_ftrs[0], self.var_out_ftrs[0])
        self.linear_map1 = nn.Linear(self.var_out_ftrs[0], 1)
        # discrete var 2:
        self.embed2 =   nn.Embedding(self.var_out_ftrs[1], self.var_out_ftrs[1])
        self.linear_map2 = nn.Linear(self.var_out_ftrs[1], 1)
        
        # for sampling noise in each local layer
        self.batch_size = args.batch_size
        self.d = args.data_variable_size
        self.noise_dim = self.d
        '''_______________________
        '''

        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias) # W.size() = [10x100]?
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        
        # fc2: local linear layers
        layers = []
        for l in range(self.d): # for each output variable,
            # construct a local linear layer matched to its dim
            #### we need to concatenate noise at each local layer, so input_ftrs = d+1
            if self.var_out_ftrs[l] > 1:
                # discrete vars will have more than 1 'class', so turn discrete flag ON.
                layers.append(LocallyConnected(input_features=self.d+1, output_features=self.var_out_ftrs[l], bias=bias, discrete=True))
            else:
                layers.append(LocallyConnected(input_features=self.d+1, output_features=self.var_out_ftrs[l], bias=bias, discrete=False))
        self.fc2 = nn.ModuleList(layers)
        # list of 10 linear layers  

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, data):  # [n, d] -> [n, d]
        
        discrete_vars, cont_vars = data['discrete'], data['continuous']
        
        ''' map discrete variables through embedding then linear layer '''
        d1 = discrete_vars[0]
        d1 = self.linear1(self.embed1(d1)) 
        # variable is now 1-dim real --> concatenate

        d2 = discrete_vars[1]
        d2 = self.linear2(self.embed2(d2)) 
        # variable is also now 1-dim real --> concatenate

        # concatenate all variables (each 1-dim)
        x = torch.cat((d1, d2, cont_vars), dim=1)

        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]

        fake_data = []
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1] 
            # is this an activation function?
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, 1)))).double().cuda()
            x = torch.cat((x, z), dim=1)
            x = fc(x)  # [n, d, m2]
            # needs to have F.softmax() for discrete layers:
            # outputs then need to be concatenated:
            # - contin: 1-dim
            # - discre: C-dim. This means, we'll have a list of probs at the dth node that are 
            #                  then compared to the list of target probs. The loss then measures
            #                  CE between: CE(preds, target) one-hot labels. Accuracy computed
            #                  by argmax(preds), and counting how many in the batch were correctly
            #                  'classified'.
            if fc.discrete:
                xd_tilde = F.softmax(xd_tilde)
            # fake_data.append(xd_tilde)
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.
    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not
    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]
    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]


    Changes by Calum (5/10/22):
    - new attribute: line 293
        --> self.discrete: useful flag during forward(). 
                           means we can call F.softmax() if discrete=True, otherwise not.

    """

    # def __init__(self, num_linear, input_features, output_features, bias=True):
    def __init__(self, input_features, output_features, bias=True, discrete=False):
        super(LocallyConnected, self).__init__()
        # self.num_linear = num_linear # dont need this, since we construct 1 at a time now
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.discrete = discrete

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )

class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().cpu().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input.cuda()

trace_expm = TraceExpm.apply
    
class AAE_WGAN_GP(nn.Module):
    """DAG-AAE model/framework."""
    def __init__(self, args, adj_A):
        super(AAE_WGAN_GP, self).__init__()
        
        self.data_type = args.data_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        
        self.discriminator_steps = args.discriminator_steps
        self.epochs = args.epochs
        self.lr = args.lr
        
        self.c_A = args.c_A
        self.lambda_A = args.lambda_A
        self.tau_A = args.tau_A
        self.graph_threshold = args.graph_threshold
        
        self.x_dims = args.x_dims
        self.z_dims = args.z_dims
        self.encoder_hidden = args.encoder_hidden
        self.decoder_hidden = args.decoder_hidden
        self.adj_A = adj_A
        
        self.k_max_iter = int(args.k_max_iter)
        self.h_tol = args.h_tol
        
        self.h_A_new = torch.tensor(1.)
        self.h_A_old = np.inf
        
        self.discrete_columns = args.discrete_column_names_list
        self.data_variable_size = self.adj_A.shape[1]
        
        self.lr_decay = args.lr_decay
        self.gamma = args.gamma
        self.negative_slope = args.negative_slope

    def forward(self, inputs):
        ''' in this case, inputs are X; noise (z) comes in at each local layer.
            But, X contains mixed data. In its current form, X will have 
            already been processed so that integer labels are assigned to each
            discrete variable. 
            Before entering fc1 of generator, these integers need to be 
            transformed into continuous repres. (via embed+linear).
        '''
        # z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.data_variable_size)))).double().to(self.device)
        
        fake_data = self.generator(inputs)

        return fake_data
        #en_outputs, logits, new_adjA, Wa = self.encoder(inputs)
        #mat_z, de_outputs = self.decoder(logits, new_adjA, Wa)
        #return en_outputs, logits, new_adjA, mat_z, de_outputs
                
    def update_optimizer(self, optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr
        
    def vae_loss(self, data, de_outputs, logits, origin_A, tau_A, data_variable_size, lambda_A, c_A, device):
        
        target = data
        preds = de_outputs
        variance = 0.
        
        # reconstruction accuracy loss
        if self.data_type == 'benchmark':
            loss_nll = nll_catogrical(preds, target)
        else:   
            loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A 
        sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))

        # compute h(A)
        h_A = _h_A(origin_A, data_variable_size, device)
        
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss
        
        return loss, preds, target, loss_nll, loss_kl 
    
    def train(self, train_loader, epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizerG, optimizerD):
        '''training algorithm for a single epoch'''
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        self.schedulerG.step()
        self.schedulerD.step()

        # update optimizer
        optimizerG, lr = self.update_optimizer(optimizerG, self.lr, c_A)
        optimizerD, lr = self.update_optimizer(optimizerD, self.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):
            for n in range(self.discriminator_steps):
                ###################################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###################################################################
                
                data, relations = Variable(data.to(self.device)).double(), Variable(relations.to(self.device)).double()
                
                if self.data_type != 'synthetic':
                    data = data.unsqueeze(2)
                
                optimizerD.zero_grad()
                
                fake_data = self(data)
                
                y_fake = self.discriminator(fake_data)
            
                y_real = self.discriminator(data)
                
                if self.x_dims > 1:
                    #vector case
                    pen = self.discriminator.calc_gradient_penalty(
                        data.view(-1, data.size(1) * data.size(2)), fake_data.view(-1, fake_data.size(1) * fake_data.size(2)), self.data_type, self.device) 
                    loss_d = -(torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake)))
                else:
                    #normal continious and discrete data case
                    pen = self.discriminator.calc_gradient_penalty(
                            data, fake_data, self.data_type, self.device) 
                    loss_d = -(torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake)))
                    
                pen.backward(retain_graph=True)
                loss_d.backward()
                loss_d = optimizerD.step() 
            
            # ###################################################
            # # (2) Update G network which is the decoder of VAE
            # ###################################################
            
            # data, relations = Variable(data.to(self.device)).double(), Variable(relations.to(self.device)).double()
            
            # optimizerV.zero_grad()
            
            # en_outputs, logits, origin_A, mat_z, de_outputs = self(data)
            
            # loss, preds, target, loss_nll, loss_kl = self.vae_loss(data, de_outputs, logits, origin_A, self.tau_A, self.data_variable_size, lambda_A, c_A, self.device)

            # loss.backward(retain_graph=True)
            # loss = optimizerV.step()
            
            # # compute metrics
            # graph = origin_A.data.clone().cpu().numpy()
            # graph[np.abs(graph) < self.graph_threshold] = 0
            
            # if ground_truth_G != None:
            #     fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            #     shd_trian.append(shd)
                
            # mse_train.append(F.mse_loss(preds, target).item())
            # nll_train.append(loss_nll.item())
            # kl_train.append(loss_kl.item())
            
            ###############################################
            # (3) Update G network: maximize log(D(G(z)))
            ###############################################

            optimizerG.zero_grad()
            
            fake_data = self(data)
            
            y_fake = self.discriminator(fake_data) #cloning is absolutely necessary here
            
            loss_g = -torch.mean(y_fake)
            
            h_A = self.generator.h_func()
            
            loss_g += lambda_A * h_A + 0.5 * c_A * h_A * h_A 
            
            loss_g.backward()
            loss_g = optimizerG.step() 
            
            # compute metrics
            #graph = origin_A.data.clone().cpu().numpy()
            graph = self.generator.fc1_to_adj()
            graph[np.abs(graph) < self.graph_threshold] = 0
            
            if ground_truth_G != None:
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
                shd_trian.append(shd)
            
        if ground_truth_G != None:
            
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                  'time: {:.4f}s'.format(time.time() - t))

            return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph#, origin_A
        else:
            
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'time: {:.4f}s'.format(time.time() - t))
            
            return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph#, origin_A
        
    
    def fit(self, train_loader, ground_truth_G = None):
        
        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(self.data_variable_size, (256, 256), self.negative_slope).double().to(self.device)
            
        if not hasattr(self, "generator"):
            self.generator = Generator(dims=[self.data_variable_size, 10, 1], bias=True).double().to(self.device)
            
        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
            
        if not hasattr(self, "optimizerG"):
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr)
            
        if not hasattr(self, "schedulerG"):
            self.schedulerG = lr_scheduler.StepLR(self.optimizerG, step_size=self.lr_decay, gamma=self.gamma)
            
        if not hasattr(self, "schedulerD"):
            self.schedulerD = lr_scheduler.StepLR(self.optimizerD, step_size=self.lr_decay, gamma=self.gamma)

        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_epoch = 0
        best_ELBO_graph = []
        best_NLL_graph = []
        best_MSE_graph = []

        try:
            for step_k in range(self.k_max_iter):
                while self.c_A < 1e+20:
                    for epoch in range(self.epochs):
                        #ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = self.train(train_loader,
                        #epoch, best_ELBO_loss, ground_truth_G, 
                        #self.lambda_A, self.c_A, self.optimizerV, self.optimizerD)
                        ELBO_loss, NLL_loss, MSE_loss, graph = self.train(train_loader,
                        epoch, best_ELBO_loss, ground_truth_G, 
                        self.lambda_A, self.c_A, self.optimizerG, self.optimizerD)
                        if ELBO_loss < best_ELBO_loss:
                            best_ELBO_loss = ELBO_loss
                            best_epoch = epoch
                            best_ELBO_graph = graph

                        if NLL_loss < best_NLL_loss:
                            best_NLL_loss = NLL_loss
                            best_epoch = epoch
                            best_NLL_graph = graph

                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch
                            best_MSE_graph = graph

                    print("Optimization Finished!")
                    print("Best Epoch: {:04d}".format(best_epoch))
                
                    if ELBO_loss > 2 * best_ELBO_loss:
                        break

                    # # update parameters
                    # A_new = origin_A.data.clone()
                    # self.h_A_new = _h_A(A_new, self.data_variable_size, self.device)
                    # if self.h_A_new.item() > 0.25 * self.h_A_old:
                    #     self.c_A*=10
                    # else:
                    #     break
                    with torch.no_grad():
                        self.h_A_new = self.generator.h_func().item()
                    if self.h_A_new > 0.25 * self.h_A_old:
                        self.c_A*=10
                    else:
                        break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                #self.h_A_old = self.h_A_new.item()
                #self.lambda_A += self.c_A * self.h_A_new.item()
                self.h_A_old = self.h_A_new
                self.lambda_A += self.c_A * self.h_A_new

                # if self.h_A_new.item() <= self.h_tol:
                #     break
                
                if self.h_A_new <= self.h_tol:
                     break
                
            if ground_truth_G != None:
                # test()
                #print (best_ELBO_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
                print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #print(best_NLL_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
                print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #print (best_MSE_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
                print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #graph = origin_A.data.clone().cpu().numpy()
                graph = self.generator.fc1_to_adj()
                graph[np.abs(graph) < 0.1] = 0
                # print(graph)
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
                print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                graph[np.abs(graph) < 0.2] = 0
                # print(graph)
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
                print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                graph[np.abs(graph) < 0.3] = 0
                # print(graph)
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
                print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
                
                #graph = origin_A.data.clone().cpu().numpy()
                #graph[np.abs(graph) < self.graph_threshold] = 0
                return graph
            else:
                #graph = origin_A.data.clone().cpu().numpy()
                graph = self.generator.fc1_to_adj()
                graph[np.abs(graph) < self.graph_threshold] = 0
                return graph

        except KeyboardInterrupt:
            # print the best anway
            #print(best_ELBO_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
            print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            #print(best_NLL_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
            print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            #print(best_MSE_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
            print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            #graph = origin_A.data.clone().cpu().numpy()
            graph = self.generator.fc1_to_adj()
            graph[np.abs(graph) < 0.1] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            graph[np.abs(graph) < 0.2] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            graph[np.abs(graph) < 0.3] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    
    def save_model(self):
        assert self.save_directory != '', 'Saving directory not specified! Please specify a saving directory!'
        torch.save(self.encoder.state_dict(), os.path.join(self.save_directory,'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_directory,'decoder.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_directory,'discriminator.pth'))
        
    def load_model(self):
        assert self.load_directory != '', 'Loading directory not specified! Please specify a loading directory!'
        
        encoder = MLPEncoder(self.x_dims, self.encoder_hidden, int(self.z_dims), self.adj_A,
                                  self.device, self.data_type).double().to(self.device)
            
        decoder = MLPDecoder(self.z_dims, self.x_dims, self.decoder_hidden,
                                  self.device, self.data_type).double().to(self.device)
            
        discriminator = Discriminator(self.data_variable_size, (256, 256)).double().to(self.device)
            
            
        encoder.load_state_dict(torch.load(os.path.join(self.load_directory,'encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(self.load_directory,'decoder.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(self.load_directory,'discriminator.pth')))
            
            
        return encoder, decoder, discriminator