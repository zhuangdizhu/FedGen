import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

class LogitTracker():
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels) # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels,unique_labels) )

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels) )
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_


    def avg(self):
        res= self.logit_sums / self.label_counts.float().unsqueeze(1)
        return res


class UserFedDistill(User):
    """
    Track and average logit vectors for each label, and share it with server/other users.
    """
    def __init__(self, args, id, model, train_data, test_data, unique_labels, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.init_loss_fn()
        self.unique_labels = unique_labels
        self.label_counts = {}
        self.logit_tracker = LogitTracker(self.unique_labels)
        self.global_logits = None
        self.reg_alpha = 1

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=True, lr_decay=True, count_labels=True, verbose=True):
        self.clean_up_counts()
        self.model.train()
        REG_LOSS, TRAIN_LOSS = 0, 0
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])
                self.optimizer.zero_grad()
                result=self.model(X, logit=True)
                output, logit = result['output'], result['logit']
                self.logit_tracker.update(logit, y)
                if self.global_logits != None:
                    ### get desired logit for each sample
                    train_loss = self.loss(output, y)
                    target_p = F.softmax(self.global_logits[y,:], dim=1)
                    reg_loss = self.ensemble_loss(output, target_p)
                    REG_LOSS += reg_loss
                    TRAIN_LOSS += train_loss
                    loss = train_loss + self.reg_alpha * reg_loss
                else:
                    loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.local_model)
            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
        if self.global_logits != None and verbose:
            REG_LOSS = REG_LOSS.detach().numpy() / (self.local_epochs * self.K)
            TRAIN_LOSS = TRAIN_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = "Train loss {:.2f}, Regularization loss {:.2f}".format(REG_LOSS, TRAIN_LOSS)
            print(info)



