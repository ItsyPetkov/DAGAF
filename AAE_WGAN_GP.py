# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:37:53 2021
@author: Hristo Petkov
"""

import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as sopt
import numpy as np
import networkx as nx
import scipy.linalg as slin
import os
import pandas as pd

from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import Linear, Sequential, LeakyReLU, Dropout, BatchNorm1d
from torch.utils.data import DataLoader
from Utils import plot_history


class MLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_layers=1, n_units=100):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units

        # create layers
        layers = [nn.Linear(n_inputs, n_units)]
        for _ in range(n_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(n_units, n_units))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_units, n_outputs))
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layers(x)
        return x

class Discriminator(nn.Module):
    """Discriminator module."""

    def __init__(self, input_dim, discriminator_dim, negative_slope, dropout_rate, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim

        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(negative_slope), Dropout(dropout_rate)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def calc_gradient_penalty(self, real_data, fake_data, device="cpu", pac=10, lambda_=10):

        # reshape data
        real_data = real_data.squeeze()
        fake_data = fake_data.squeeze()

        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))

class Generator(nn.Module):
    def __init__(self, dims,  step=1, bias=True):
        super(Generator, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.step = step

        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()

        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            if self.step == 1:
                layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
            else:
                layers.append(LocallyConnected(d, dims[l + 1]+1, dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.init_weights()

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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            if self.step == 2:
                z=Variable(torch.FloatTensor(np.random.normal(0, 1, (x.shape[0], x.shape[1], 1))))
                x = torch.cat((x, z), dim=2)
            x = fc(x)  # [n, d, m2]
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
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

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
            self.num_linear, self.input_features, self.output_features,
            self.bias is not None
        )

class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply

class AAE_WGAN_GP(nn.Module):
    """DAG-AAE model/framework."""

    def __init__(self, args, adj_A):
        super(AAE_WGAN_GP, self).__init__()

        self.verbose = args.verbose
        self.data_type = args.data_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size

        self.discriminator_steps = args.discriminator_steps
        self.csl_steps = args.csl_steps
        self.epochs = args.epochs
        self.epochs2 = args.epochs2
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

        self.h_A_new = torch.tensor(1.0)
        self.h_A_old = np.inf

        self.discrete_columns = args.discrete_column_names_list
        self.data_variable_size = self.adj_A.shape[1]

        self.mul1 = args.mul1
        self.mul2 = args.mul2
        self.alpha = args.alpha
        self.pnl = args.pnl

        self.lr_decay = args.lr_decay
        self.gamma = args.gamma
        self.negative_slope = args.negative_slope
        self.dropout_rate = args.dropout_rate

        self.save_directory = args.save_directory
        self.load_directory = args.load_directory
        self.graph_linear_type = args.graph_linear_type
        self.settings = args.settings

        self.model = Generator(dims=[self.data_variable_size,10,1], bias=True)
        self.mlp_inverse = MLP(n_inputs=self.data_variable_size, n_outputs=self.data_variable_size, n_layers=3, n_units=self.data_variable_size*10)
        self.mlp = MLP(n_inputs=self.data_variable_size, n_outputs=self.data_variable_size, n_layers=3, n_units=self.data_variable_size*10)
        self.discriminator = Discriminator(self.data_variable_size, (256, 256), self.negative_slope, self.dropout_rate)
        self.discriminator1 = Discriminator(self.data_variable_size, (256, 256), self.negative_slope, self.dropout_rate)
        self.generator = Generator(dims=[self.data_variable_size,10,1], step=2, bias=True)
        self.step = 1

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return (1 - self.alpha) * mmd
    
    def kl_gaussian_sem(self, logits):
        mu = logits
        kl_div = mu * mu
        kl_sum = kl_div.sum()
        return (kl_sum / (logits.size(0))) * 0.5

    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss
    
    def dual_ascent_step(self, model, discriminator, generator, discriminator1, mlp_inverse, mlp, X, lambda1, lambda2, rho, alpha, h, rho_max, best_epoch, best_shd, best_mse_loss, best_shd_graph, dis1, dis2, gen, ground_truth=None):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        optimizerMLPI = optim.Adam(mlp_inverse.parameters(), lr=self.lr)
        optimizerD = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerG = optim.Adam(generator.fc2.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerD1 = optim.Adam(discriminator1.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerMLP = optim.Adam(mlp.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        
        while rho < rho_max:
            for epoch in range(self.epochs):
                t = time.time()
                for batch_idx, (data, relations) in enumerate(X):
                    for i in range(self.csl_steps):
                        ###################################################################
                        # (1) Learn causal structure with modified Notears-MLP model
                        ###################################################################
                        for n in range(self.discriminator_steps):
                            ###################################################################
                            # (1.1) Update D network: minimize ||D(x) - D(G(z))|| + GP
                            ###################################################################
                            data, relations = (Variable(data.squeeze()), Variable(relations))
                            optimizerD.zero_grad()
                            X_hat = model(data)
                            y_fake = discriminator(X_hat)
                            y_real = discriminator(data)
                            pen = discriminator.calc_gradient_penalty(data, X_hat)
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                            pen.backward(retain_graph=True)
                            loss_d.backward()
                            optimizerD.step()
                        ########################################################################
                        # (1.2) Update Notears-MLP network parameters with MSE + regularization
                        ########################################################################
                        data, relations = (Variable(data.squeeze()), Variable(relations))
                        optimizer.zero_grad()
                        X_hat = model(data)
                        kld_loss = self.kl_gaussian_sem(X_hat)
                        mmd_loss = self.compute_mmd(X_hat, data)  
                        loss = self.squared_loss(X_hat, data) + kld_loss + mmd_loss
                        h_val = model.h_func()
                        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                        l2_reg = 0.5 * lambda2 * model.l2_reg()
                        l1_reg = lambda1 * model.fc1_l1_reg()
                        primal_obj = loss + penalty + l2_reg + l1_reg
                        primal_obj.backward(retain_graph=True)
                        optimizer.step()
                        ###############################################
                        # (1.3) Update G network: maximize -D(G(z))
                        ###############################################
                        optimizer.zero_grad()
                        y_fake = discriminator(X_hat.data.clone())
                        loss_g = -torch.mean(y_fake)
                        loss_g.backward()
                        optimizer.step()
                        ##################################################################################
                        # (1.4) Update the inverted MLP network parameters with MSE (for PNL models only)
                        ##################################################################################
                        if self.pnl:
                            optimizerMLPI.zero_grad()
                            optimizer.zero_grad()
                            Y_hat = mlp_inverse(data)
                            pnl_loss = self.squared_loss(X_hat, Y_hat)
                            pnl_loss.backward()
                            optimizerMLPI.step()
                            optimizer.step()
                        
                    ###################################################################
                    # (2) Produce diverse samples using the WGAN architecture
                    ###################################################################
                    with torch.no_grad():
                        generator.fc1_pos.weight.copy_(model.fc1_pos.weight)
                        generator.fc1_neg.weight.copy_(model.fc1_neg.weight)
                        self.step = 2

                    if self.pnl:
                        for n in range(self.discriminator_steps):
                            ###################################################################
                            # (2.1) Update D network: minimize ||D(x) - D(G(z))|| + GP
                            ###################################################################
                            data, relations = (Variable(data.squeeze()), Variable(relations))
                            optimizerD1.zero_grad()
                            X_hat = generator(data)
                            X_tilde = mlp(X_hat)
                            y_fake = discriminator1(X_tilde)
                            y_real = discriminator1(data)
                            pen = discriminator1.calc_gradient_penalty(data, X_tilde)
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                            if epoch % 100 == 0:
                                dis2.append(torch.mean(y_fake).clone().detach().numpy())
                                dis1.append(torch.mean(y_real).clone().detach().numpy())
                            pen.backward(retain_graph=True)
                            loss_d.backward()
                            optimizerD1.step()
                        ###############################################
                        # (2.2) Update G network: maximize -D(G(z))
                        ###############################################
                        data, relations = (Variable(data.squeeze()), Variable(relations))
                        optimizerG.zero_grad()
                        optimizerMLP.zero_grad()
                        X_hat = generator(data)
                        X_tilde = mlp(X_hat)
                        y_fake = discriminator1(X_tilde)
                        loss_g = -torch.mean(y_fake)
                        if epoch % 100 == 0:
                            gen.append(loss_g.clone().detach().numpy())
                        loss_g.backward()
                        optimizerG.step()
                        optimizerMLP.step()
                        self.step = 1
                    else:
                        for n in range(self.discriminator_steps):
                            ###################################################################
                            # (2.1) Update D network: minimize ||D(x) - D(G(z))|| + GP
                            ###################################################################
                            data, relations = (Variable(data.squeeze()), Variable(relations))
                            optimizerD1.zero_grad()
                            X_hat = generator(data)
                            y_fake = discriminator1(X_hat)
                            y_real = discriminator1(data)
                            pen = discriminator1.calc_gradient_penalty(data, X_hat)
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                            if epoch % 100 == 0:
                                dis2.append(torch.mean(y_fake).clone().detach().numpy())
                                dis1.append(torch.mean(y_real).clone().detach().numpy())
                            pen.backward(retain_graph=True)
                            loss_d.backward()
                            optimizerD1.step()
                        ###############################################
                        # (2.2) Update G network: maximize -D(G(z))
                        ###############################################
                        data, relations = (Variable(data.squeeze()), Variable(relations))
                        optimizerG.zero_grad()
                        X_hat = generator(data)
                        y_fake = discriminator1(X_hat)
                        loss_g = -torch.mean(y_fake)
                        if epoch % 100 == 0:
                            gen.append(loss_g.clone().detach().numpy())
                        loss_g.backward()
                        optimizerG.step()
                        self.step = 1
                
                W_est = model.fc1_to_adj()
                W_est[np.abs(W_est) < 0.3] = 0
                acc = self.count_accuracy(ground_truth, W_est != 0)

                if self.pnl:
                    if best_shd == np.inf and best_mse_loss == np.inf and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_shd = acc['shd']
                        best_mse_loss = self.squared_loss(X_tilde, data).item()
                    elif acc['shd'] < best_shd and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_shd = acc['shd']
                        best_epoch = epoch
                        best_shd_graph = W_est
                        np.save("causal_graph.npy", best_shd_graph) 
                        best_mse_loss = self.squared_loss(X_tilde, data).item()
                    elif acc['shd'] == best_shd and self.squared_loss(X_tilde, data).item() < best_mse_loss and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_mse_loss = self.squared_loss(X_tilde, data).item()
                        best_epoch = epoch
                        self.save_model()
                else:
                    if best_shd == np.inf and best_mse_loss == np.inf and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_shd = acc['shd']
                        best_mse_loss = self.squared_loss(X_hat, data).item()
                    elif acc['shd'] < best_shd and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_shd = acc['shd']
                        best_epoch = epoch
                        best_shd_graph = W_est
                        np.save("causal_graph.npy", best_shd_graph)  
                        best_mse_loss = self.squared_loss(X_hat, data).item()
                    elif acc['shd'] == best_shd and self.squared_loss(X_hat, data).item() < best_mse_loss and nx.is_directed_acyclic_graph(nx.DiGraph(W_est)):
                        best_mse_loss = self.squared_loss(X_hat, data).item()
                        best_epoch = epoch
                        self.save_model() 

                if self.verbose:
                    if ground_truth is not None:
                        print(
                            "Step: {:01d}".format(self.step), 
                            "Epoch: {:04d}".format(epoch),
                            "D_loss: {:.10f}".format(-np.mean(loss_d.item())),
                            "G_loss: {:.10f}".format(-np.mean(loss_g.item())),
                            "Pen_loss: {:.10f}".format(np.mean(pen.item())),
                            "Kld_loss: {:.10f}".format(np.mean(kld_loss.item())),
                            "Mmd_loss: {:.10f}".format(np.mean(mmd_loss.item())),
                            "mse_train: {:.10f}".format(np.mean(self.squared_loss(X_hat, data).item())),
                            "shd_trian: {:.10f}".format(np.mean(acc['shd'])),
                            "time: {:.4f}s".format(time.time() - t),
                        )
                    else:
                        print(
                            "Step: {:01d}".format(self.step), 
                            "Epoch: {:04d}".format(epoch),
                            "D_loss: {:.10f}".format(-np.mean(loss_d.item())),
                            "G_loss: {:.10f}".format(-np.mean(loss_g.item())),
                            "Pen_loss: {:.10f}".format(np.mean(pen.item())),
                            "Kld_loss: {:.10f}".format(np.mean(kld_loss.item())),
                            "Mmd_loss: {:.10f}".format(np.mean(mmd_loss.item())),
                            "mse_train: {:.10f}".format(np.mean(self.squared_loss(X_hat, data).item())),
                            "time: {:.4f}s".format(time.time() - t),
                        )

            if self.verbose:
                if ground_truth is not None:
                    print("Optimization Finished!")
                    print("Best Epoch: {:04d}".format(best_epoch))
                    if best_shd == np.inf:
                        print("Best SHD: inf")
                    else:
                        print("Best SHD: {:04d}".format(best_shd))
                    print("Best MSE Loss: {:.10f}".format(best_mse_loss))
                else:
                    print("Optimization Finished!")
                    print("Best Epoch: {:04d}".format(best_epoch))
                    print("Best MSE Loss: {:.10f}".format(best_mse_loss))

            with torch.no_grad():
                h_new = model.h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new, best_shd, best_epoch, best_shd_graph, best_mse_loss

    def notears_nonlinear(self, model: nn.Module,
                        discriminator: nn.Module,
                        generator: nn.Module,
                        discriminator1: nn.Module,
                        mlp_inverse: nn.Module,
                        mlp: nn.Module,
                        X: np.ndarray,
                        ground_truth: np.ndarray = None,
                        lambda1: float = 0.,
                        lambda2: float = 0.,
                        max_iter: int = 100,
                        h_tol: float = 1e-8,
                        rho_max: float = 1e+16,
                        w_threshold: float = 0.3):
        rho, alpha, h = 1.0, 0.0, np.inf
        best_mse_loss, best_shd, best_epoch, best_shd_graph  = np.inf, np.inf, 0, []
        dis1, dis2, gen = [], [], []
        X.pin_memory_device = ''
        for _ in range(max_iter):
            rho, alpha, h, best_shd, best_epoch, best_shd_graph, best_mse_loss = self.dual_ascent_step(model, discriminator, generator, discriminator1, mlp_inverse, mlp, X, lambda1, lambda2,
                                            rho, alpha, h, rho_max, best_epoch, best_shd, best_mse_loss, best_shd_graph, dis1, dis2, gen, ground_truth)
            if h <= h_tol or rho >= rho_max:
                break
        best_shd_graph[np.abs(best_shd_graph) < w_threshold] = 0
        plot_history(dis1, dis2, gen)
        return best_shd_graph

    def fit(self, model, discriminator, generator, discriminator1, mlp_inverse, mlp, train_data, ground_truth):
        if self.settings == "EP":
            causal_graph = self.notears_nonlinear(model, discriminator, generator, discriminator1, mlp_inverse, mlp, train_data, ground_truth, lambda1=0.01, lambda2=0.01)
            real_df, fake_df = self.sample(train_data, causal_graph)
            return causal_graph, real_df, fake_df
        elif self.settings == "CSL":
            causal_graph = self.notears_nonlinear(model, discriminator, generator, discriminator1, mlp_inverse, mlp, train_data, ground_truth, lambda1=0.01, lambda2=0.01)
            return causal_graph
        elif self.settings == "DG":
            real_df, fake_df = self.sample(train_data)
            return real_df, fake_df
        
    
    def generateSyntheticData(self, dims, generator, causal_graph, graph_threshold=0.05, device="cpu"):

        batch_size, n_nodes, z_dim = dims[0], dims[1], dims[2]

        params = [p for p in generator.fc2.parameters()]
        weights, biases = params[0], params[1]

        G = nx.DiGraph(causal_graph)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == n_nodes
        print("Ordered nodes: {}".format(ordered_vertices))

        pred_X = torch.zeros((batch_size, n_nodes)).to(device)
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, n_nodes, z_dim)))).to(device)

        with torch.no_grad():
            for i in range(n_nodes):
                current_node = int(ordered_vertices[i])
                weight = weights[current_node]
                bias = biases[current_node]

                h = generator.fc1_pos(pred_X) - generator.fc1_neg(pred_X)
                h = h.view(-1, n_nodes, n_nodes)
                h = torch.sigmoid(h)

                h_node = h[:, current_node].unsqueeze(1)
                z_node = z[:, current_node].unsqueeze(1)
                cat_hz = torch.cat((h_node, z_node), dim=2)

                out = cat_hz.unsqueeze(2) @ weight.unsqueeze(0).unsqueeze(0)
                out = out.squeeze(2)
                out += bias

                pred_X[:, current_node] = out.squeeze()

        return pred_X

    def sample(self, train_loader, causal_graph=None, columns=None):

        real_tensor_data = train_loader.dataset.tensors[0].squeeze()
        real_df = pd.DataFrame(real_tensor_data.numpy(), columns=columns)
        real_df["data"] = "real"

        n_synth = real_df.shape[0]
        num_nodes = real_df.shape[1] - 1

        generator = Generator(dims=[self.data_variable_size,10,1], step=2, bias=True)
        if causal_graph is None:
            _, generator, _, mlp, causal_graph = self.load_model()
        else:
            _, generator, _, mlp, _ = self.load_model()
        generator.eval()
        mlp.eval()

        synthetic_data = self.generateSyntheticData(dims=[n_synth, num_nodes, 1], generator=generator, causal_graph=causal_graph)

        if self.pnl:
            with torch.no_grad():
                synthetic_data = mlp(synthetic_data)

        fake_df = pd.DataFrame(synthetic_data.cpu().numpy(), columns=columns)
        fake_df["data"] = "fake"

        return real_df, fake_df
    
    def count_accuracy(self, B_true, B_est):
        """Compute various accuracy metrics for B_est.

        true positive = predicted association exists in condition in correct direction
        reverse = predicted association exists in condition in opposite direction
        false positive = predicted association does not exist in condition

        Args:
            B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
            B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

        Returns:
            fdr: (reverse + false positive) / prediction positive
            tpr: (true positive) / condition positive
            fpr: (reverse + false positive) / condition negative
            shd: undirected extra + undirected missing + reverse
            nnz: prediction positive
        """
        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
        d = B_true.shape[0]
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
    
    def save_model(self):
        assert (self.save_directory != ""), "Saving directory not specified! Please specify a saving directory!"
        torch.save(self.model.state_dict(), os.path.join(self.save_directory, "model.pth"))
        torch.save(self.generator.state_dict(), os.path.join(self.save_directory, "generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_directory, "discriminator.pth"))
        torch.save(self.mlp.state_dict(), os.path.join(self.save_directory, "MLP.pth"))

    def load_model(self):
        assert (self.load_directory != ""), "Loading directory not specified! Please specify a loading directory!"

        model = Generator(dims=[self.data_variable_size,10,1], bias=True)
        discriminator = Discriminator(self.data_variable_size, (256, 256), self.negative_slope, self.dropout_rate)
        generator = Generator(dims=[self.data_variable_size,10,1], step=2, bias=True)
        mlp = MLP(n_inputs=self.data_variable_size, n_outputs=self.data_variable_size, n_layers=3, n_units=self.data_variable_size*10)
        causal_graph = np.load("causal_graph.npy")

        model.load_state_dict(torch.load(os.path.join(self.load_directory, "model.pth")))
        generator.load_state_dict(torch.load(os.path.join(self.load_directory, "generator.pth")))
        discriminator.load_state_dict(torch.load(os.path.join(self.load_directory, "discriminator.pth")))
        mlp.load_state_dict(torch.load(os.path.join(self.load_directory, "MLP.pth")))

        return model, generator, discriminator, mlp, causal_graph


    
