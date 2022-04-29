from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGE_SIZE = 84
# TODO change LOAD_PATH to be your own data path
LOAD_PATH = './celeba/'
DUMP_PATH = './'
IMG_DIR = os.path.join(LOAD_PATH, 'data/raw/img_align_celeba')
random.seed(42)
np.random.seed(42)

def load_proxy_data(user_lists, cdata):
    n_users = len(user_lists)
    # merge all uer data chosen for proxy
    new_data={
        'classes': [],
        'user_data': {
            'x': [],
            'y': []
        },
        'num_samples': 0
    }
    for uname in user_lists:
        user_data = cdata['user_data'][uname]
        X=user_data['x'] # path to image
        y=user_data['y'] # label
        assert len(X) == len(y)
        ## load image ##
        loaded_X = []
        for i, image_name in enumerate(X):
            image_path = os.path.join(IMG_DIR, image_name)
            image = Image.open(image_path)
            image=image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
            image = np.array(image)
            loaded_X.append(image)

        new_data['user_data']['x'] += np.array(loaded_X).tolist()
        new_data['user_data']['y'] += y

        new_data['classes'] += list(np.unique(y))
        new_data['num_samples'] += len(y)
    combined = list(zip(new_data['user_data']['x'], new_data['user_data']['y']))
    return new_data


def load_data(user_lists, cdata, agg_user=-1):
    n_users = len(user_lists)
    new_data = {
        'users': [None for _ in range(n_users)],
        'num_samples': [None for _ in range(n_users)],
        'user_data': {}
    }
    if agg_user > 0:
        assert len(user_lists) % agg_user == 0
        agg_n_users = len(user_lists) // agg_user
        agg_data = {
            'users': [None for _ in range(agg_n_users)],
            'num_samples': [None for _ in range(agg_n_users)],
            'user_data': {}

        }

    def agg_by_user_(new_data, agg_n_users, agg_user, verbose=False):
        for batch_id in range(agg_n_users):
            start_id, end_id = batch_id * agg_user, (batch_id + 1) * agg_user
            X, Y, N_samples = [], [], 0
            for idx in range(start_id, end_id):
                user_uname='f_{0:05d}'.format(idx)
                x = new_data['user_data'][user_uname]['x']
                y = new_data['user_data'][user_uname]['y']
                n_samples = new_data['num_samples'][idx]
                X += x
                Y += y
                N_samples += n_samples

            #####
            batch_user_name = 'f_{0:05d}'.format(batch_id)
            agg_data['users'][batch_id]= batch_user_name
            agg_data['num_samples'][batch_id]=len(Y)
            agg_data['user_data'][batch_user_name]={
                'x': torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2),
                'y': torch.Tensor(Y).type(torch.int64)
            }
            #####


    def load_per_user_(user_data, idx, verbose=False):
        """
        # Reduce test samples per user to ratio.
        :param uname:
        :param idx:
        :return:
        """
        new_uname='f_{0:05d}'.format(idx)
        X=user_data['x']
        y=user_data['y']
        assert len(X) == len(y)
        new_data['users'][idx] = new_uname
        new_data['num_samples'][idx] = len(y)
        new_data['user_data'][new_uname] = {'y':y}
        # load X as images
        loaded_X = []
        for i, image_name in enumerate(X):
            image_path = os.path.join(IMG_DIR, image_name)
            image = Image.open(image_path)
            image=image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
            image = np.array(image)
            loaded_X.append(image)
        new_data['user_data'][new_uname] = {'x': np.array(loaded_X).tolist(), 'y':y}
        if verbose:
            print("processing user {}".format(new_uname))

    for idx, uname in enumerate(user_lists):
        user_data = cdata['user_data'][uname]
        load_per_user_(user_data, idx, verbose=True)
        #pass
    if agg_user == -1:
        return new_data
    else:
        agg_by_user_(new_data, agg_n_users, agg_user, verbose=False)
        return agg_data

def process_data():
    load_path =  os.path.join(LOAD_PATH, 'data')
    train_files = [f for f in os.listdir(os.path.join(load_path, 'train'))]
    test_files = [f for f in os.listdir(os.path.join(load_path, 'test'))]

    def sample_users(cdata, ratio=0.1, excludes=set()):
        """
        :param cdata:
        :param ratio:
        :return: list of sampled user names
        """
        user_lists=[u for u in cdata['users']]
        if  ratio <= 1:
            n_selected_users=int(len(user_lists) * ratio)
        else:
            n_selected_users = ratio
        random.shuffle(user_lists)
        new_users = []
        i = 0
        for u in user_lists:
            if u not in excludes:
                new_users.append(u)
                i += 1
            if i == n_selected_users:
                return new_users


    def process_(mode, tf, ratio=0.1, user_lists=None, agg_user=-1):
        read_path = os.path.join(load_path, mode if mode != 'proxy' else 'train', tf)
        with open(read_path, 'r') as inf:
            cdata=json.load(inf)
            n_users = len(cdata['users'])
            if ratio > 1:
                assert ratio < n_users
            else:
                assert ratio < 1
            print("Number of users: {}".format(n_users))
            print("Number of raw {} samples: {:.1f}".format(mode, np.mean(cdata['num_samples'])))
            print("Deviation of raw {} samples: {:.1f}".format(mode, np.std(cdata['num_samples'])))
            #exit()
            if mode == 'train':
                assert user_lists == None
                user_lists = sample_users(cdata, ratio)
                new_data=load_data(user_lists, cdata, agg_user=agg_user)
            else: # test mode
                assert len(user_lists) > 0
                new_data = load_data(user_lists, cdata, agg_user=agg_user)
            print("Number of reduced users: {}".format(len(new_data['num_samples'])))
            print("Number of samples per user: {}".format(new_data['num_samples']))

        if ratio > 1:
            n_users = int(ratio)
            if agg_user > 0:
                n_users = int( n_users // agg_user)
        else:
            n_users = int(len(cdata['users']) * ratio)
        if agg_user > 0:
            dump_path=os.path.join(DUMP_PATH, 'user{}-agg{}'.format(n_users, agg_user))
        else:
            dump_path=os.path.join(DUMP_PATH, 'user{}'.format(n_users))
        os.system("mkdir -p {}".format(os.path.join(dump_path, mode)))

        dump_path = os.path.join(dump_path, '{}/{}.pt'.format(mode,mode))
        with open(dump_path, 'wb') as outfile:
           print("Saving {} data to {}".format(mode, dump_path))
           torch.save(new_data, outfile)
        return user_lists

    #
    mode='train'
    tf = train_files[0]
    user_lists = process_(mode, tf, ratio=args.ratio, agg_user=args.agg_user)
    mode = 'test'
    tf = test_files[0]
    process_(mode, tf, ratio=args.ratio, user_lists=user_lists, agg_user=args.agg_user)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg_user", type=int, default=10, help="number of celebrities to be aggregated together as a device/client (as meta-batch size).")
    parser.add_argument("--ratio", type=float, default=250, help="Number of total celebrities to be sampled for FL training.")
    args = parser.parse_args()
    print("Number of FL devices: {}".format(args.ratio // args.agg_user ))
    process_data()