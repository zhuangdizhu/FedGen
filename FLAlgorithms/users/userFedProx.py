import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer

class UserFedProx(User):
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, lr_decay=True, count_labels=False):
        self.clean_up_counts()
        self.model.train()
        # cache global model initialized value to local model
        self.clone_model_paramenter(self.local_model, self.model.parameters())
        for epoch in range(self.local_epochs):
            self.model.train()
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                output=self.model(X)['output']
                loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.local_model)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
