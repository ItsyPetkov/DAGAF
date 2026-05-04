# Shell script for running multiple experiments
# Works for Additive Noise Model and Post-NonLinear Model
# 5 iterations where the same data will be used as input for the follwoing models:
# DAG-NOTEARS, DAG-NOTAERS-MLP, DAG-GNN, GAE, DAG-WGAN, DAG-WGAN+ and MRC-GAN

# PNL-1 experiments with 10 nodes
graph_linear_type="post_nonlinear_1" 
data_variable_size=10
verbose=0

myfunc()
{
    # 5 iterations
    for (( c=0; c<5; c++ ))
    do
        # MRC-GAN
        echo "MRC-GAN training"
        python Main.py --synthesize 1 --pnl 0 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --export_directory ./ --verbose $verbose
        echo " "

        echo "MRC-GAN training"
        python Main.py --synthesize 0 --pnl 1 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --export_directory ./ --verbose $verbose
        echo " "

        # # MRC-GAN
        # echo "MRC-GAN training"
        # python Main.py --synthesize 0 --alpha 1 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --verbose $verbose
        # echo " "

        # # MRC-GAN
        # echo "MRC-GAN training"
        # python Main.py --synthesize 0 --alpha 2 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --verbose $verbose
        # echo " "

        # # MRC-GAN
        # echo "MRC-GAN training"
        # python Main.py --synthesize 0 --alpha 5 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --verbose $verbose
        # echo " "

        # # MRC-GAN
        # echo "MRC-GAN training"
        # python Main.py --synthesize 0 --alpha 10 --data_variable_size $data_variable_size --graph_linear_type $graph_linear_type --save_directory ./ --load_directory ./ --verbose $verbose
        # echo " "

        # # DAG-NOTEARS and DAG-NOTEARS-MLP
        # # Copying files to the correct directory
        # cp ./train_loader.pkl ~/code/WGANCI/WGAN_Tabular/notears-master/notears/
        # cp ./ground_truth_G.pkl ~/code/WGANCI/WGAN_Tabular/notears-master/notears/

        # # Navigate to the model directory
        # cd ~/code/WGANCI/WGAN_Tabular/notears-master/notears/

        # # Running both scripts
        # echo "DAG-NOTEARS training"
        # python linear.py
        # echo " "

        # echo "DAG-NOTEARS-MLP training"
        # python nonlinear.py --data_variable_size $data_variable_size
        # echo " "

        # # DAG-GNN
        # # Copying files to the correct directory
        # cp ./train_loader.pkl ~/code/WGANCI/WGAN_Tabular/OG-DAG-GNN/DAG-GNN-master/src/
        # cp ./ground_truth_G.pkl ~/code/WGANCI/WGAN_Tabular/OG-DAG-GNN/DAG-GNN-master/src/

        # # Navigate to the model directory
        # cd ~/code/WGANCI/WGAN_Tabular/OG-DAG-GNN/DAG-GNN-master/src/

        # # Running the script
        # echo "DAG-GNN training"
        # python train.py --verbose $verbose
        # echo " "

        # # DAG-WGAN
        # # Copying files to the correct directory
        # cp ./train_loader.pkl ~/git/git/DAG-WGAN/DAG-WGAN/
        # cp ./ground_truth_G.pkl ~/git/git/DAG-WGAN/DAG-WGAN/

        # # Navigate to the model directory
        # cd ~/git/git/DAG-WGAN/DAG-WGAN/

        # # Running the script
        # echo "DAG-WGAN training"
        # python Main.py --data_variable_size $data_variable_size --save_directory ./ --verbose $verbose
        # echo " "

        # # DAG-WGAN+
        # # Copying files to the correct directory
        # cp ./train_loader.pkl ~/code/WGANCI/WGAN_Tabular/New_DAG_WGAN/
        # cp ./ground_truth_G.pkl ~/code/WGANCI/WGAN_Tabular/New_DAG_WGAN/

        # # Navigate to the model directory
        # cd ~/code/WGANCI/WGAN_Tabular/New_DAG_WGAN/

        # # Running the script
        # echo "DAG-WGAN+ training"
        # python Main.py --data_variable_size $data_variable_size --save_directory ./ --verbose $verbose
        # echo " "

        #Navigate back to the original directory
        cd ~/git/git/OG-MRC-GAN/
    done

    #cd ~/git/git/OG-MRC-GAN/
}

myfunc $graph_linear_type $data_variable_size $verbose