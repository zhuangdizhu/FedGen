from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
# Implementation for FedProx Server

class FedProx(Server):
    def __init__(self, args, model, seed):
        #dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
        #         local_epochs, num_users, K, personal_learning_rate, times):
        super().__init__(args, model, seed)#dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         #local_epochs, num_users, times)

        # Initialize data for all  users
        data = read_data(args.dataset)
        total_users = len(data[0])
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserFedProx(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", self.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters()
            self.evaluate()
            for user in self.selected_users: # allow selected users to train
                    user.train(glob_iter)
            self.aggregate_parameters()
        self.save_results(args)
        self.save_model()