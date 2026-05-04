set data_variable_size=10
set graph_linear_type="nonlinear_2"
set save_directory="C:\Users\PC\git\MRC-GAN_New\train_logs"

ECHO Begining Synthetic Experiments with %data_variable_size% variables and %graph_linear_type% SEM > .\results.txt
ECHO: >> .\results.txt

ECHO First Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Second Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size%>> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Third Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fourth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fifth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt
@REM cd .\..\..\..\MRC-GAN_NEW\

set data_variable_size=20

ECHO: >> .\results.txt
ECHO Begining Synthetic Experiments with %data_variable_size% variables and %graph_linear_type% SEM >> .\results.txt
ECHO: >> .\results.txt

ECHO First Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Second Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Third Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fourth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fifth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt
@REM cd .\..\..\..\MRC-GAN_NEW\

set data_variable_size=50

ECHO: >> .\results.txt
ECHO Begining Synthetic Experiments with %data_variable_size% variables and %graph_linear_type% SEM >> .\results.txt
ECHO: >> .\results.txt

ECHO First Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Second Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Third Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fourth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fifth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt
@REM cd .\..\..\..\MRC-GAN_NEW\

set data_variable_size=100

ECHO: >> .\results.txt
ECHO Begining Synthetic Experiments with %data_variable_size% variables and %graph_linear_type% SEM >> .\results.txt
ECHO: >> .\results.txt

ECHO First Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Second Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Third Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fourth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
ECHO: >> .\results.txt
ECHO Fifth Synthetic Experiment >> .\results.txt
ECHO: >> .\results.txt
ECHO Notears-GAN >> .\results.txt
python .\Main.py --graph_linear_type %graph_linear_type% --data_variable_size %data_variable_size% --verbose 0 --save_directory %save_directory% --synthesize 1 >> .\results.txt

@REM ECHO: >> .\results.txt
@REM ECHO DAG-Notears >> .\results.txt
@REM copy .\ground_truth_G.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\ground_truth_G.pkl
@REM copy .\train_loader.pkl .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\train_loader.pkl
@REM cd .\..\..\.spyder-py3\PhDCode\WGANCI\WGAN_Tabular\notears-master\notears\
@REM python .\linear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-Notears-MLP >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM python .\nonlinear.py --data_variable_size %data_variable_size% >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM ECHO DAG-WGAN >> .\..\..\..\..\..\..\git\MRC-GAN_NEW\results.txt
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\ground_truth_G.pkl
@REM copy .\..\..\..\..\..\..\git\MRC-GAN_NEW\train_loader.pkl .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\train_loader.pkl
@REM cd .\..\..\..\..\..\..\git\DAG-WGAN\DAG-WGAN\
@REM python .\Main.py --data_variable_size %data_variable_size% >> .\..\..\MRC-GAN_NEW\results.txt

@REM ECHO: >> .\..\..\MRC-GAN_NEW\results.txt
@REM ECHO DAG-GNN >> .\..\..\MRC-GAN_NEW\results.txt
@REM copy .\..\..\MRC-GAN_NEW\ground_truth_G.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\ground_truth_G.pkl
@REM copy .\..\..\MRC-GAN_NEW\train_loader.pkl .\..\..\OG-DAG-GNN\DAG-GNN-master\src\train_loader.pkl
@REM cd .\..\..\OG-DAG-GNN\DAG-GNN-master\src\
@REM python .\train.py --data_variable_size %data_variable_size% >> .\..\..\..\MRC-GAN_NEW\results.txt

@REM cd .\..\..\..\MRC-GAN_NEW\
@REM start code .\results.txt @REM in case you prefer vsCode
start notepad .\results.txt
@REM pause
@REM del .\results.txt