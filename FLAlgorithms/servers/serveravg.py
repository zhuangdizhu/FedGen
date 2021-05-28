from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedAvg Server
import time

class FedAvg(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all  users
        data = read_data(args.dataset)
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserAVG(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters(mode=self.mode)
            self.evaluate()
            self.timestamp = time.time() # log user-training start time
            for user in self.selected_users: # allow selected users to train
                    user.train(glob_iter, personalized=self.personalized) #* user.train_samples
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time
            self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        self.save_results(args)
        self.save_model()