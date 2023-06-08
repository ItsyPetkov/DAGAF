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
from Utils import preprocess_adj_new, preprocess_adj_new1
from Utils import nll_gaussian, kl_gaussian_sem, nll_catogrical
from Utils import _h_A
from Utils import count_accuracy


class Discriminator(nn.Module):
    """Discriminator module."""

    def __init__(
        self, input_dim, discriminator_dim, negative_slope, dropout_rate, pac=10
    ):
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

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=10, lambda_=10
    ):

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
    def __init__(self, n, m, dims, device, bias=True):
        super(Generator, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.d = dims[0]
        self.dims = dims
        self.m = m
        self.n = n
        self.device = device
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d, self.d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(self.d, self.d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(
                LocallyConnected(self.d, dims[l + 1] + self.m, dims[l + 2], bias=bias)
            )
        self.fc2 = nn.ModuleList(layers)
        self.init_weights()

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
            z = (
                Variable(
                    torch.FloatTensor(np.random.normal(0, 1, (self.n, self.d, self.m)))
                )
                .double()
                .to(self.device)
            )
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
        reg = 0.0
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

        self.weight = nn.Parameter(
            torch.Tensor(num_linear, input_features, output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

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
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear, self.in_features, self.out_features, self.bias is not None
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input.to(device)


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

        self.lr_decay = args.lr_decay
        self.gamma = args.gamma
        self.negative_slope = args.negative_slope
        self.dropout_rate = args.dropout_rate

        self.save_directory = args.save_directory
        self.load_directory = args.load_directory

        self.step = 1

    def forward(self, inputs):
        if self.step == 1:
            fake_data = self.mlp(inputs.squeeze())
        else:
            fake_data = self.generator(inputs.squeeze())
        return fake_data

    def update_optimizer(self, optimizer, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
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
            parame_group["lr"] = lr

        return optimizer, lr

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

    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    def train(
        self,
        train_loader,
        epoch,
        best_epoch,
        best_mse_loss,
        best_mse_data,
        best_shd,
        best_shd_graph,
        ground_truth_G,
        lambda_A,
        c_A,
        optimizerD_csl,
        optimizerMLP,
        optimizerG,
        optimizerD,
    ):
        """training algorithm for a single epoch"""
        t = time.time()
        D_loss = []
        G_loss = []
        Pen_loss = []
        MMD_loss = []
        mse_train = []
        shd_trian = []

        # update optimizers
        optimizerD_csl, lr = self.update_optimizer(optimizerD_csl, self.lr, c_A)
        optimizerMLP, lr = self.update_optimizer(optimizerMLP, self.lr, c_A)
        optimizerG, lr = self.update_optimizer(optimizerG, self.lr, c_A)
        optimizerD, lr = self.update_optimizer(optimizerD, self.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):

            ###################################################################
            # (1) Learn causal structure with original Notears-MLP model
            ###################################################################

            for i in range(self.csl_steps):
                for n in range(self.discriminator_steps):

                    data, relations = (
                        Variable(data.to(self.device)).double(),
                        Variable(relations.to(self.device)).double(),
                    )

                    if self.data_type != "synthetic":
                        data = data.unsqueeze(2)

                    optimizerD_csl.zero_grad()

                    fake_data = self(data)

                    y_fake = self.discriminator_csl(fake_data)

                    y_real = self.discriminator_csl(data)

                    if self.x_dims > 1:
                        # vector case
                        pen = self.discriminator_csl.calc_gradient_penalty(
                            data.view(-1, data.size(1) * data.size(2)),
                            fake_data.view(-1, fake_data.size(1) * fake_data.size(2)),
                            self.device,
                        )
                        loss_d = -(
                            torch.mean(F.softplus(y_real))
                            - torch.mean(F.softplus(y_fake))
                        )
                    else:
                        # normal continious and discrete data case
                        pen = self.discriminator_csl.calc_gradient_penalty(
                            data, fake_data, self.device
                        )
                        loss_d = -(
                            torch.mean(F.softplus(y_real))
                            - torch.mean(F.softplus(y_fake))
                        )

                    Pen_loss.append(pen.item())
                    pen.backward(retain_graph=True)

                    D_loss.append(loss_d.item())
                    loss_d.backward()
                    loss_d = optimizerD_csl.step()

                data, relations = (
                    Variable(data.to(self.device)).double(),
                    Variable(relations.to(self.device)).double(),
                )

                optimizerMLP.zero_grad()

                fake_data = self(data)

                loss_mmd = self.compute_mmd(fake_data, data.squeeze())

                MMD_loss.append(loss_mmd.item())

                loss = self.squared_loss(fake_data, data.squeeze()) + loss_mmd

                h_A = self.mlp.h_func()

                penalty = lambda_A * h_A + 0.5 * c_A * h_A * h_A

                l2_reg = 0.5 * self.mul2 * self.mlp.l2_reg()
                l1_reg = self.mul1 * self.mlp.fc1_l1_reg()

                loss_mlp = loss + penalty + l2_reg + l1_reg

                loss_mlp.backward(retain_graph=True)
                loss_mlp = optimizerMLP.step()

            ###################################################################
            # (2) Produce diverse samples using the WGAN architecture
            ###################################################################

            self.step = 2

            with torch.no_grad():
                self.generator.fc1_pos.weight.copy_(self.mlp.fc1_pos.weight)
                self.generator.fc1_neg.weight.copy_(self.mlp.fc1_neg.weight)

            for n in range(self.discriminator_steps):

                data, relations = (
                    Variable(data.to(self.device)).double(),
                    Variable(relations.to(self.device)).double(),
                )

                optimizerD.zero_grad()

                fake_data_gen = self(data)

                y_fake = self.discriminator(fake_data_gen)

                y_real = self.discriminator(data)

                if self.x_dims > 1:
                    # vector case
                    pen = self.discriminator.calc_gradient_penalty(
                        data.view(-1, data.size(1) * data.size(2)),
                        fake_data_gen.view(
                            -1, fake_data_gen.size(1) * fake_data_gen.size(2)
                        ),
                        self.device,
                    )
                    loss_d = -(
                        torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake))
                    )
                else:
                    # normal continious and discrete data case
                    pen = self.discriminator.calc_gradient_penalty(
                        data, fake_data_gen, self.device
                    )
                    loss_d = -(
                        torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake))
                    )

                Pen_loss.append(pen.item())
                pen.backward(retain_graph=True)

                D_loss.append(loss_d.item())
                loss_d.backward()
                loss_d = optimizerD.step()

            optimizerG.zero_grad()

            data, relations = (
                Variable(data.to(self.device)).double(),
                Variable(relations.to(self.device)).double(),
            )

            fake_data_gen = self(data)

            y_fake = self.discriminator(fake_data_gen)

            loss_g = -torch.mean(F.softplus(y_fake))

            G_loss.append(loss_g.item())
            loss_g.backward()

            loss_g = optimizerG.step()

            self.step = 1

            self.schedulerD_csl.step()
            self.schedulerMLP.step()
            self.schedulerG.step()
            self.schedulerD.step()

            # compute metrics
            graph = self.mlp.fc1_to_adj()
            graph[np.abs(graph) < self.graph_threshold] = 0

            mse = F.mse_loss(fake_data, data.squeeze()).item()

            if ground_truth_G != None:
                fdr, tpr, fpr, shd, nnz = count_accuracy(
                    ground_truth_G, nx.DiGraph(graph)
                )
                shd_trian.append(shd)

                if best_shd == np.inf and best_mse_loss == np.inf:
                    best_shd = shd
                    best_mse_loss = mse
                elif shd < best_shd:
                    best_shd = shd
                    best_shd_graph = graph
                    best_mse_loss = np.inf
                elif shd == best_shd and mse < best_mse_loss:
                    best_mse_loss = mse
                    best_mse_data = fake_data
                    best_epoch = epoch
                    self.save_model()
            else:
                if best_mse_loss == np.inf:
                    best_mse_loss = mse
                elif mse < best_mse_loss:
                    best_mse_loss = mse
                    best_mse_data = fake_data

            mse_train.append(mse)

        if ground_truth_G != None:
            if self.verbose:
                print(
                    "Epoch: {:04d}".format(epoch),
                    "D_loss: {:.10f}".format(np.mean(D_loss)),
                    "G_loss: {:.10f}".format(np.mean(G_loss)),
                    "Pen_loss: {:.10f}".format(np.mean(Pen_loss)),
                    "MMD_loss: {:.10f}".format(np.mean(MMD_loss)),
                    "mse_train: {:.10f}".format(np.mean(mse_train)),
                    "shd_trian: {:.10f}".format(np.mean(shd_trian)),
                    "time: {:.4f}s".format(time.time() - t),
                )

            return (
                np.mean(D_loss),
                np.mean(G_loss),
                np.mean(Pen_loss),
                np.mean(MMD_loss),
                np.mean(mse_train),
                graph,
                best_shd,
                best_shd_graph,
                best_mse_loss,
                best_mse_data,
                best_epoch,
            )  # , origin_A
        else:
            if self.verbose:
                print(
                    "Epoch: {:04d}".format(epoch),
                    "D_loss: {:.10f}".format(np.mean(D_loss)),
                    "G_loss: {:.10f}".format(np.mean(G_loss)),
                    "Pen_loss: {:.10f}".format(np.mean(Pen_loss)),
                    "MMD_loss: {:.10f}".format(np.mean(MMD_loss)),
                    "mse_train: {:.10f}".format(np.mean(mse_train)),
                    "time: {:.4f}s".format(time.time() - t),
                )

            return (
                np.mean(D_loss),
                np.mean(G_loss),
                np.mean(Pen_loss),
                np.mean(MMD_loss),
                np.mean(mse_train),
                graph,
                best_shd,
                best_shd_graph,
                best_mse_loss,
                best_mse_data,
                best_epoch,
            )  # , origin_A

    def fit(self, train_loader, ground_truth_G=None):

        if not hasattr(self, "discriminator_csl"):
            self.discriminator_csl = (
                Discriminator(
                    self.data_variable_size,
                    (256, 256),
                    self.negative_slope,
                    self.dropout_rate,
                )
                .double()
                .to(self.device)
            )

        if not hasattr(self, "mlp"):
            self.mlp = (
                Generator(
                    self.batch_size,
                    self.z_dims,
                    dims=[self.data_variable_size, 10, 1],
                    device=self.device,
                    bias=True,
                )
                .double()
                .to(self.device)
            )

        if not hasattr(self, "discriminator"):
            self.discriminator = (
                Discriminator(
                    self.data_variable_size,
                    (256, 256),
                    self.negative_slope,
                    self.dropout_rate,
                )
                .double()
                .to(self.device)
            )

        if not hasattr(self, "generator"):
            self.generator = (
                Generator(
                    self.batch_size,
                    self.z_dims,
                    dims=[self.data_variable_size, 10, 1],
                    device=self.device,
                    bias=True,
                )
                .double()
                .to(self.device)
            )

        if not hasattr(self, "optimizerD_csl"):
            self.optimizerD_csl = optim.Adam(
                self.discriminator_csl.parameters(),
                lr=self.lr,
                betas=(0.5, 0.9),
                weight_decay=1e-6,
            )

        if not hasattr(self, "optimizerMLP"):
            self.optimizerMLP = optim.Adam(self.mlp.parameters(), lr=self.lr)

        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(
                self.discriminator.parameters(),
                lr=self.lr,
                betas=(0.5, 0.9),
                weight_decay=1e-6,
            )

        if not hasattr(self, "optimizerG"):
            self.optimizerG = optim.Adam(self.generator.fc2.parameters(), lr=self.lr)

        if not hasattr(self, "schedulerD_csl"):
            self.schedulerD_csl = lr_scheduler.StepLR(
                self.optimizerD_csl, step_size=self.lr_decay, gamma=self.gamma
            )

        if not hasattr(self, "schedulerMLP"):
            self.schedulerMLP = lr_scheduler.StepLR(
                self.optimizerMLP, step_size=self.lr_decay, gamma=self.gamma
            )

        if not hasattr(self, "schedulerD"):
            self.schedulerD = lr_scheduler.StepLR(
                self.optimizerD, step_size=self.lr_decay, gamma=self.gamma
            )

        if not hasattr(self, "schedulerG"):
            self.schedulerG = lr_scheduler.StepLR(
                self.optimizerG, step_size=self.lr_decay, gamma=self.gamma
            )

        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_shd = np.inf
        best_epoch = 0
        best_ELBO_graph = []
        best_NLL_graph = []
        best_MSE_graph = []
        best_shd_graph = []
        best_MSE_data = []

        try:
            for step_k in range(self.k_max_iter):
                while self.c_A < 1e20:
                    for epoch in range(self.epochs):
                        (
                            D_loss,
                            G_loss,
                            Pen_loss,
                            Info_loss,
                            MSE_loss,
                            graph,
                            best_shd,
                            best_shd_graph,
                            best_MSE_loss,
                            best_MSE_data,
                            best_epoch,
                        ) = self.train(
                            train_loader,
                            epoch,
                            best_epoch,
                            best_MSE_loss,
                            best_MSE_data,
                            best_shd,
                            best_shd_graph,
                            ground_truth_G,
                            self.lambda_A,
                            self.c_A,
                            self.optimizerD_csl,
                            self.optimizerMLP,
                            self.optimizerG,
                            self.optimizerD,
                        )

                    if self.verbose:
                        if ground_truth_G != None:
                            print("Optimization Finished!")
                            print("Best Epoch: {:04d}".format(best_epoch))
                            print("Best SHD: {:04d}".format(best_shd))
                            print("Best MSE Loss: {:.10f}".format(best_MSE_loss))
                        else:
                            print("Optimization Finished!")
                            print("Best Epoch: {:04d}".format(best_epoch))
                            print("Best MSE Loss: {:.10f}".format(best_MSE_loss))
                    # print("Best SHD Graph:")
                    # print(best_shd_graph)
                    # print("Best MSE Data:")
                    # print(best_MSE_data)

                    # update parameters
                    with torch.no_grad():
                        self.h_A_new = self.generator.h_func().item()
                    if self.h_A_new > 0.25 * self.h_A_old:
                        self.c_A *= 10
                    else:
                        break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                self.h_A_old = self.h_A_new
                self.lambda_A += self.c_A * self.h_A_new

                if self.h_A_new <= self.h_tol:
                    break

            if ground_truth_G != None:

                best_shd_graph[np.abs(best_shd_graph) < 0.3] = 0
                # print(graph)
                fdr, tpr, fpr, shd, nnz = count_accuracy(
                    ground_truth_G, nx.DiGraph(best_shd_graph)
                )
                print(
                    "threshold 0.3, Accuracy: fdr",
                    fdr,
                    " tpr ",
                    tpr,
                    " fpr ",
                    fpr,
                    "shd",
                    shd,
                    "nnz",
                    nnz,
                )

                return best_shd_graph, best_MSE_data
            else:
                # graph = self.generator.fc1_to_adj()
                graph[np.abs(graph) < self.graph_threshold] = 0
                return graph, best_MSE_data

        except KeyboardInterrupt:
            # print the best anway
            # print(best_ELBO_graph)
            # print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_ELBO_graph)
            )
            print(
                "Best ELBO Graph Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

            # print(best_NLL_graph)
            # print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_NLL_graph)
            )
            print(
                "Best NLL Graph Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

            # print(best_MSE_graph)
            # print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_MSE_graph)
            )
            print(
                "Best MSE Graph Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

            # best_shd_graph = self.generator.fc1_to_adj()
            best_shd_graph[np.abs(best_shd_graph) < 0.1] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_shd_graph)
            )
            print(
                "threshold 0.1, Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

            best_shd_graph[np.abs(best_shd_graph) < 0.2] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_shd_graph)
            )
            print(
                "threshold 0.2, Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

            best_shd_graph[np.abs(best_shd_graph) < 0.3] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(
                ground_truth_G, nx.DiGraph(best_shd_graph)
            )
            print(
                "threshold 0.3, Accuracy: fdr",
                fdr,
                " tpr ",
                tpr,
                " fpr ",
                fpr,
                "shd",
                shd,
                "nnz",
                nnz,
            )

    def generateSyntheticData(
        self, dims, generator, var_names, graph_threshold=0.05, device="cuda"
    ):

        batch_size, n_nodes, z_dim = dims[0], dims[1], dims[2]

        generator.eval()
        params = [p for p in generator.fc2.parameters()]
        weights, biases = params[0], params[1]

        A = generator.fc1_to_adj()
        A[np.abs(A) < graph_threshold] = 0
        G = nx.DiGraph(A)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == n_nodes
        print("Ordered nodes: {}".format(ordered_vertices))

        pred_X = torch.zeros((batch_size, n_nodes)).double().to(device)
        z = Variable(
            torch.DoubleTensor(np.random.normal(0, 1, (batch_size, n_nodes, z_dim)))
        ).to(device)

        weights_dict, biases_dict = {}, {}

        with torch.no_grad():
            for i in range(n_nodes):
                current_node = int(ordered_vertices[i])
                weight = weights[current_node]
                bias = biases[current_node]
                # weights_dict[var_names[current_node]] = [w.detach().cpu().numpy() for w in weight]
                # biases_dict[var_names[current_node]] = [b.detach().cpu().numpy() for b in biases]

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

    def sample(self, train_loader, columns=None):

        real_tensor_data = train_loader.dataset.tensors[0].squeeze()
        real_df = pd.DataFrame(real_tensor_data.numpy(), columns=columns)
        real_df["data"] = "real"

        weights = "./generator.pth"
        n_synth = real_df.shape[0]
        num_nodes = real_df.shape[1] - 1

        generator = (
            Generator(
                n_synth,
                1,
                dims=[num_nodes, num_nodes, 1],
                device=self.device,
                bias=True,
            )
            .double()
            .to(self.device)
        )
        generator.load_state_dict(torch.load(weights))
        generator.eval()

        synthetic_data = self.generateSyntheticData(
            dims=[n_synth, num_nodes, 1], generator=generator
        )

        fake_df = pd.DataFrame(synthetic_data.cpu().numpy(), columns=columns)
        fake_df["data"] = "fake"

        return fake_df

    def save_model(self):
        assert (
            self.save_directory != ""
        ), "Saving directory not specified! Please specify a saving directory!"
        torch.save(
            self.discriminator_csl.state_dict(),
            os.path.join(self.save_directory, "discriminator_csl.pth"),
        )
        torch.save(
            self.mlp.state_dict(), os.path.join(self.save_directory, "MLP.pth"),
        )
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.save_directory, "generator.pth"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(self.save_directory, "discriminator.pth"),
        )

    def load_model(self):
        assert (
            self.load_directory != ""
        ), "Loading directory not specified! Please specify a loading directory!"

        discriminator_csl = (
            Discriminator(
                self.data_variable_size,
                (256, 256),
                self.negative_slope,
                self.dropout_rate,
            )
            .double()
            .to(self.device)
        )

        mlp = (
            Generator(
                self.batch_size,
                self.z_dims,
                self.x_dims,
                dims=[self.data_variable_size, 10, 1],
                device=self.device,
                bias=True,
            )
            .double()
            .to(self.device)
        )

        generator = (
            Generator(
                self.batch_size,
                self.z_dims,
                self.x_dims,
                dims=[self.data_variable_size, 10, 1],
                device=self.device,
                step=2,
                bias=True,
            )
            .double()
            .to(self.device)
        )
        discriminator = (
            Discriminator(
                self.data_variable_size,
                (256, 256),
                self.negative_slope,
                self.dropout_rate,
            )
            .double()
            .to(self.device)
        )

        discriminator_csl.load_state_dict(
            torch.load(os.path.join(self.load_directory, "discriminator_csl.pth"))
        )
        mlp.load_state_dict(torch.load(os.path.join(self.load_directory, "MLP.pth")))
        generator.load_state_dict(
            torch.load(os.path.join(self.load_directory, "generator.pth"))
        )
        discriminator.load_state_dict(
            torch.load(os.path.join(self.load_directory, "discriminator.pth"))
        )

        return discriminator_csl, mlp, generator, discriminator
