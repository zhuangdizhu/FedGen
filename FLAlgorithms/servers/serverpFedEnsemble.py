from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_test_data
import torch.nn as nn
import numpy as np

class FedEnsemble(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.slow_start = 20
        self.use_adam = 'adam' in self.algorithm.lower()
        self.init_ensemble_configs()
        self.init_loss_fn()
        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info =read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            user=UserAVG(args, id, model, train_data, test_data, use_adam=self.use_adam)
            self.users.append(user)

        #### build test data loader ####
        self.testloaderfull, self.unique_labels=aggregate_user_test_data(data, args.dataset, self.total_test_samples)
        print("Loading testing data.")
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            self.send_parameters()
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                user.train(
                    glob_iter,
                    personalized=False, lr_decay=True, count_labels=True)
            self.aggregate_parameters()
            self.evaluate_ensemble(selected=False)

        self.save_results(args)
        self.save_model()
