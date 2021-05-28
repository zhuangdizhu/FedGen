#!/bin/sh
prefix="/mnt/research-share/projects/zhuzhuan/"
if [ "$#" -le 2 ]
then
    echo "Usage:      ./batch_run_jobs.sh <data> <<alpha> <n_seeds> <if_run>
    echo "Example:     ./batch_run_jobs.sh emnist 0.1 3 1
    exit
else
    data=$1
    alpha=$2
    times=$3
    running=$4
    ### shared parameters ###
    lamda="1"
    model="cnn"
    local_epochs=20
    batch_size=32
    num_users=10
    num_glob_iters=200
    learning_rate=0.01
    #################################
    # EMnist data
    #################################
    if [ "$data" = "emnist" ]; then
      ratio=0.1
      for alg in FedAvg FedGen FedProx FedEnsemble FedDistill-FL
      do
        dataset="EMnist-alpha$alpha-ratio$ratio"
        cmd="python3 main.py --dataset $dataset  --algorithm $alg --batch_size $batch_size --local_epochs $local_epochs --num_users $num_users --lamda $lamda --model $model --learning_rate $learning_rate  --num_glob_iters $num_glob_iters --times $times --K 1"
        echo $cmd
      if [ "$running" = "1" ]; then
        eval $cmd
      fi
    done

    #################################
    # Mnist data
    #################################
    elif [ "$data" = "mnist" ]; then
      ratio=0.5
      for alg in FedAvg FedGen FedProx FedEnsemble FedDistill-FL
      do
        dataset="Mnist-alpha$alpha-ratio$ratio"
        cmd="python3 main.py --dataset $dataset  --algorithm $alg --batch_size $batch_size --local_epochs $local_epochs --num_users $num_users --lamda $lamda --model $model --learning_rate $learning_rate  --num_glob_iters $num_glob_iters --times $times --K 1"
        echo $cmd
      if [ "$running" = "1" ]; then
        eval $cmd
      fi
    done

    else
      extra_cmd=""
    fi
fi