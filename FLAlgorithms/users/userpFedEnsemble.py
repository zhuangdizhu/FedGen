import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import numpy as np
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from torchvision.utils import save_image

class UserpFedEnsemble(User):
    def __init__(self, dataset, algorithm, numeric_id, train_data, test_data,
                 model, generative_model, available_labels,
                 batch_size, learning_rate, beta, lamda, local_epochs, K):
        super().__init__(dataset, algorithm, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs)

        self.init_loss_fn()
        self.K = K
        self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.generative_model = copy.deepcopy(generative_model)
        self.label_counts = {}
        self.available_labels = available_labels
        self.generative_alpha = 10
        self.generative_beta = 0.1
        self.update_gen_freq = 5
        self.pretrained = False
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=1e-3, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.generative_optimizer, gamma=0.98)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def update_generator(self, steps, verbose=True):
        self.model.eval()
        self.generative_model.train()
        RECONSTRUCT_LOSS, KLD_LOSS, RC_LOSS = 0, 0, 0
        for _ in range(steps):
            self.generative_optimizer.zero_grad()
            samples=self.get_next_train_batch(count_labels=True)
            X, y=samples['X'], samples['y']
            gen_result = self.generative_model(X, y)
            loss_info = self.generative_model.loss_function(
                gen_result['output'],
                X,
                gen_result['mu'],
                gen_result['log_var'],
                beta=0.01
            )
            loss, kld_loss, reconstruct_loss = loss_info['loss'], loss_info['KLD'], loss_info['reconstruction_loss']
            RECONSTRUCT_LOSS += loss
            KLD_LOSS += kld_loss
            RC_LOSS += reconstruct_loss
            loss.backward()
            self.generative_optimizer.step()
        self.generative_lr_scheduler.step()

        if verbose:
            RECONSTRUCT_LOSS = RECONSTRUCT_LOSS.detach().numpy() / steps
            KLD_LOSS = KLD_LOSS.detach().numpy() / steps
            RC_LOSS = RC_LOSS.detach().numpy() / steps
            info = "VAE-Loss: {:.4f}, KL-Loss: {:.4f}, RC-Loss:{:.4f}".format(RECONSTRUCT_LOSS, KLD_LOSS, RC_LOSS)
            print(info)


    def train(self, glob_iter, personalized=False, reconstruct=False, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, RECONSTRUCT_LOSS = 0, 0, 0
        #if glob_iter % self.update_gen_freq == 0:
        if not self.pretrained:
            self.update_generator(self.local_epochs * 20)
            self.pretrained = True
            self.visualize_images(self.generative_model, 0, repeats=10)
        for epoch in range(self.local_epochs):
            self.model.train()
            self.generative_model.eval()
            for i in range(self.K):
                loss = 0
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                samples =self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                self.update_label_counts(samples['labels'], samples['counts'])
                model_result=self.model(X, return_latent=reconstruct)
                output = model_result['output']
                predictive_loss=self.loss(output, y)
                loss += predictive_loss
                #### sample from generator and regulate Dist|z_gen, z_pred|, where z_gen = Gen(x, y), z_pred = model(X)
                if reconstruct:
                    gen_result = self.generative_model(X, y, latent=True)
                    z_gen = gen_result['latent']
                    z_model = model_result['latent']
                    dist_loss = self.generative_beta * self.dist_loss(z_model, z_gen)
                    DIST_LOSS += dist_loss
                    loss += dist_loss
                #### get loss and perform optimization
                loss.backward()
                self.optimizer.step()  # self.local_model)

        if reconstruct:
            DIST_LOSS+=(torch.mean(DIST_LOSS.double())).item()
        # local-model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        if reconstruct and verbose:
            info = 'Latent Reconstruction Loss={:.4f}'.format(DIST_LOSS)
            print(info)

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-user{self.id}-iter{glob_iter}.png'
        y=self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input=torch.tensor(y, dtype=torch.int64)
        generator.eval()
        images=generator.sample(y_input, latent=False)['output'] # 0,1,..,K, 0,1,...,K
        images=images.view(repeats, -1, *images.shape[1:])
        images=images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))