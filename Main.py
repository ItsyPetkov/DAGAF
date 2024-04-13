# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:49:31 2020
@author: Hristo Petkov
"""

# Importing libraries and frameworks
import time
import os
import torch
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from AAE_WGAN_GP import AAE_WGAN_GP
from Utils import load_data
from Utils import draw_dag
from Utils import compute_BiCScore
from argparse import ArgumentParser

# Adding a lot of args here
parser = ArgumentParser()
parser.add_argument("--path", type=str, default="", help="choosing a path for the input.")
parser.add_argument("--column_names_list", type=str, nargs="+", default=[], help="choosing the column names for samping of original dataframe")
parser.add_argument("--discrete_column_names_list", type=str, nargs="+", default=[], help="choosing the discrete column names in the dataframe")
parser.add_argument("--discriminator_steps", type=int, default=1, help="Number of steps for the discriminator")
parser.add_argument("--csl_steps", type=int, default=1, help="Number of steps for the causal structure learning")
parser.add_argument("--initial_identifier", type=str, default="", help="Initial Identifier for the sample dataframe")
parser.add_argument("--num_of_rows", type=int, default=-1, help="Number of rows in the sampled dataframe")
parser.add_argument("--save_directory", default="", type=str, help="A directory to save a trained model to.")
parser.add_argument("--load_directory", default="", type=str, help="A directory to load a trained model from.")
parser.add_argument("--export_directory", type=str, default="", help="choosing a directory for the output.")
parser.add_argument("--verbose", type=int, default=1, help="used to control the print statements per epoch.")

# -----------data parameters ------
# configurations
parser.add_argument("--synthesize", type=int, default=0, help="Flag for synthesing synthetic data")
parser.add_argument("--pnl", type=int, default=0, help="Flag for Post-Non-Linear model") # otherwise Additive Noise Model is assumed
parser.add_argument("--data_type", type=str, default="synthetic", choices=["synthetic", "benchmark", "real"], help="choosing which experiment to do.")
parser.add_argument("--data_sample_size", type=int, default=5000, help="the number of samples of data")
parser.add_argument("--data_variable_size", type=int, default=10, help="the number of variables in synthetic generated data")
parser.add_argument("--graph_type", type=str, default="erdos-renyi", help="the type of DAG graph by generation method")
parser.add_argument("--graph_degree", type=int, default=3, help="the number of degree in generated DAG graph")
parser.add_argument("--graph_sem_type", type=str, default="linear-gauss", help="the structure equation model (SEM) parameter type")
parser.add_argument("--graph_linear_type", type=str, default="nonlinear_2", help="the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z," 
                    + "nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z" + 'post_nonlinear_1 -> x=sinh(Acos(x+1)+z), post_nonlinear_2 -> x=tanh(2sin(A(x+0.5))+A(x+0.5)+z)')
parser.add_argument("--edge-types", type=int, default=2, help="The number of edge types to infer.")
parser.add_argument("--x_dims", type=int, default=1, help="The number of input dimensions: default 1.") # vector case: need to be equal to the last dimension of vector data to work
parser.add_argument("--z_dims", type=int, default=1, help="The number of latent variable dimensions: default the same as variable size.")
parser.add_argument("--steps", type=int, default=1, help="Number of steps for time-series data generation")
parser.add_argument("--export", type=int, default=1, help="Flag for exporting data")

# -----------training hyperparameters
parser.add_argument( "--graph_threshold", type=float, default=0.3, help="threshold for learned adjacency matrix binarization") # 0.3 is good, 0.2 is error prune
parser.add_argument("--tau_A", type=float, default=0.0, help="coefficient for L-1 norm of A.")
parser.add_argument("--lambda_A", type=float, default=0.0, help="coefficient for DAG constraint h(A).")
parser.add_argument("--c_A", type=float, default=1, help="coefficient for absolute value h(A).")
parser.add_argument("--negative_slope", type=float, default=0.2, help="negative_slope for leaky_relu")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="rate for discriminator dropout")
parser.add_argument("--noise", type=float, default=1e-20, help="amount of noise for the ANM")


parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for step 1 to train.")
parser.add_argument("--epochs2", type=int, default=2400, help="Number of epochs for step 2 to train.")
parser.add_argument("--batch_size", type=int, default=1000,  help="Number of samples per batch.") # note: should be divisible by sample size, otherwise throw an error
parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate.") # basline rate = 1e-3
parser.add_argument("--encoder-hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument("--decoder-hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument("--k_max_iter", type=int, default=1e2, help="the max iteration number for searching lambda and c")
parser.add_argument("--mul1", default=0.01, type=float, help="multiplier for the L1_Loss")
parser.add_argument("--mul2", default=0.01, type=float, help="multiplier for the L2_Loss")
parser.add_argument("--alpha", default=0, type=float, help="alpha multiplier for the MMD")

parser.add_argument("--suffix", type=str, default="_springs5", help='Suffix for training data (e.g. "_charged".')
parser.add_argument("--h_tol", type=float, default=1e-8, help="the tolerance of error of h(A) to zero")
parser.add_argument("--lr-decay", type=int, default=200, help="After how epochs to decay LR by a factor of gamma.")
parser.add_argument("--gamma", type=float, default=1.0, help="LR decay factor.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature for Gumbel softmax.")
parser.add_argument("--hard", action="store_true", default=False, help="Uses discrete samples in training forward pass.")

args = parser.parse_args()
print(args)

# controlls randomness of the entire program
torch.manual_seed(args.seed)

def main():

    t = time.time()

    if args.data_type == "real":

        train_loader, data_variable_size, columns = load_data(args, args.batch_size, args.suffix)

        # add adjacency matrix A
        num_nodes = data_variable_size
        adj_A = np.zeros((num_nodes, num_nodes))

        aae_wgan_gp = AAE_WGAN_GP(args, adj_A)

        causal_graph, data = aae_wgan_gp.fit(train_loader)

        draw_dag(causal_graph, args.data_type, columns)

    elif args.data_type == "benchmark":

        train_loader, data_variable_size, ground_truth_G, num_cats = load_data(args, args.batch_size, args.suffix)

        # add adjacency matrix A
        num_nodes = data_variable_size
        adj_A = np.zeros((num_nodes, num_nodes))

        aae_wgan_gp = AAE_WGAN_GP(args, adj_A)

        causal_graph, data = aae_wgan_gp.fit(train_loader, ground_truth_G)

        BIC_score = compute_BiCScore(np.asarray(nx.to_numpy_matrix(ground_truth_G)), causal_graph)
        print("BIC_score: " + str(BIC_score))

        # draw_dag(causal_graph, args.data_type)

    else:
        if args.synthesize:
            # create and store synthetic data
            train_loader, ground_truth_G = load_data(args, args.batch_size, args.suffix)

            with open(r"train_loader.pkl", "wb") as output_file:
                pickle.dump(train_loader, output_file)

            with open(r"ground_truth_G.pkl", "wb") as output_file_G:
                pickle.dump(ground_truth_G, output_file_G)

        # load synthetic data
        with open(r"train_loader.pkl", "rb") as input_file:
            train_data = pickle.load(input_file)

        with open(r"ground_truth_G.pkl", "rb") as input_file_G:
            ground_truth = pickle.load(input_file_G)

        # add adjacency matrix A
        num_nodes = args.data_variable_size
        adj_A = np.zeros((num_nodes, num_nodes))

        aae_wgan_gp = AAE_WGAN_GP(args, adj_A)
        causal_graph, real_df, fake_df = aae_wgan_gp.fit(aae_wgan_gp.model, aae_wgan_gp.discriminator, aae_wgan_gp.generator, aae_wgan_gp.discriminator1, aae_wgan_gp.mlp_inverse, aae_wgan_gp.mlp, train_data, nx.to_numpy_array(ground_truth))
        acc = aae_wgan_gp.count_accuracy(nx.to_numpy_array(ground_truth), causal_graph != 0)
        print("threshold 0.3, Accuracy:",acc)

    if args.export:
        assert (args.export_directory != ""), "Export directory not specified! Please specify an export directory!"
        pd.DataFrame(causal_graph).to_csv(os.path.join(args.export_directory, "adjacency_matrix.csv"), index=False)
        real_df.to_csv(os.path.join(args.export_directory, "real_data.csv"), index=False)
        fake_df.to_csv(os.path.join(args.export_directory, "generated_data.csv"), index=False)

    print("Programm finished in: "+ str(time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))

if __name__ == "__main__":
    main()
