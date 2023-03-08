# Data-Free Knowledge Distillation for Heterogeneous Federated Learning

Research code that accompanies the paper [Data-Free Knowledge Distillation for Heterogeneous Federated](https://arxiv.org/pdf/2105.10056.pdf).
It contains implementation of the following algorithms:
* **FedGen** (the proposed algorithm) ([code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverpFedGen.py)).
* **FedAvg** ([paper](https://arxiv.org/pdf/1602.05629.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serveravg.py)).
* **FedProx** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedProx.py)).
* **FedDistill** and its extension **FedDistill-FL** ([paper](https://arxiv.org/pdf/2011.02367.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedDistill.py)).

## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedGen/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
    

- Similarly, to generate *non-iid* **EMnist** Dataset, using 10% of the total available training samples:
<pre><code>cd FedGen/data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FedGen/data/EMnist/u20-letters-alpha0.1-ratio0.1/
</code></pre> 

## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the *Mnist* Dataset:
```
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedGen --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedAvg --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedProx --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedDistill-FL --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
```
----

##### Run experiments on the *EMnist* Dataset:
```
python main.py --dataset EMnist-alpha0.1-ratio0.1 --algorithm FedAvg --batch_size 32 --local_epochs 20 --num_users 10 --lamda 1 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --num_glob_iters 200 --times 3 
python main.py --dataset EMnist-alpha0.1-ratio0.1 --algorithm FedGen --batch_size 32 --local_epochs 20 --num_users 10 --lamda 1 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --num_glob_iters 200 --times 3 
python main.py --dataset EMnist-alpha0.1-ratio0.1 --algorithm FedProx --batch_size 32 --local_epochs 20 --num_users 10 --lamda 1 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --num_glob_iters 200 --times 3 
python main.py --dataset EMnist-alpha0.1-ratio0.1 --algorithm FedDistill-FL --batch_size 32 --local_epochs 20 --num_users 10 --lamda 1 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --num_glob_iters 200 --times 3 

```
----

### Plot
For the input attribute **algorithms**, list the name of algorithms and separate them by comma, e.g. `--algorithms FedAvg,FedGen,FedProx`
```
  python main_plot.py --dataset EMnist-alpha0.1-ratio0.1 --algorithms FedAvg,FedGen --batch_size 32 --local_epochs 20 --num_users 10 --num_glob_iters 200 --plot_legend 1
```
