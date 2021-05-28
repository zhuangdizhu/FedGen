import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import os
COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]

plt.rcParams.update({'font.size': 14})
n_seeds=3

def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    return name

def plot_results(args, algorithms):
    n_seeds = args.times
    dataset_ = args.dataset.split('-')
    sub_dir = dataset_[0] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
    os.system("mkdir -p figs/{}".format(sub_dir))  # e.g. figs/Mnist/ratio0.5
    plt.figure(1, figsize=(5, 5))
    TOP_N = 5
    max_acc = 0
    for i, algorithm in enumerate(algorithms):
        algo_name = get_label_name(algorithm)
        ######### plot test accuracy ############
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        all_curves = np.concatenate([metrics[seed]['glob_acc'] for seed in range(n_seeds)])
        top_accs =  np.concatenate([np.sort(metrics[seed]['glob_acc'])[-TOP_N:] for seed in range(n_seeds)] )
        acc_avg = np.mean(top_accs)
        acc_std = np.std(top_accs)
        info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
        print(info)
        length = len(all_curves) // n_seeds
        sns.lineplot(
            x=np.array(list(range(length)) * n_seeds) + 1,
            y=all_curves.astype(float),
            legend='brief',
            color=COLORS[i],
            label=algo_name,
            ci="sd",
        )

    plt.gcf()
    plt.grid()
    plt.title(dataset_[0] + ' Test Accuracy')
    plt.xlabel('Epoch')
    max_acc = np.max([max_acc, np.max(all_curves) ]) + 4e-2

    if args.min_acc < 0:
        alpha = 0.7
        min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)
    else:
        min_acc = args.min_acc
    plt.ylim(min_acc, max_acc)
    fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '.png')
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='png', dpi=400)
    print('file saved to {}'.format(fig_save_path))