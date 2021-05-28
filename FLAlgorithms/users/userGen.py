import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from utils.model_utils import get_dataset_name, CONFIGS

# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, dataset, algorithm, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, K):
        super().__init__(dataset, algorithm, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs)

        dataset_name = get_dataset_name(dataset)
        self.unique_labels = CONFIGS[dataset_name]['unique_labels']

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count


    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
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
                self.optimizer.step()#self.local_model)

            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            #self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
