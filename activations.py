import torch
import numpy as np
import matplotlib.pyplot as plt

def plot(mean, std, title):
    plt.figure(0)
    plt.vlines(np.arange(1, len(mean)+1), mean - 2*std, mean + 2*std, color='k', lw=0.5)
    plt.plot(np.arange(1, len(mean)+1), mean, color='deepskyblue', lw=1)
    plt.xlabel('Unit Indices')
    plt.ylabel('Unit Activations')
    plt.title(title)
    plt.show()


def get_hidden_activations(inputs, model, forward_intermediate):
    with torch.no_grad():
        hidden = forward_intermediate(inputs, model)
    
    return hidden.cpu().numpy()

def compute_act_stats(base_dir, output_datasets, input_dataset, p = 0.9):
    in_acts = np.loadtxt('{base_dir}/in_acts.txt'.format(base_dir=base_dir))
    avg_in_acts = np.mean(in_acts, axis=0)
    l = 0.0
    r = 1e6
    for _ in range(50):
        mid = (l+r)/2.0
        cnt = np.sum(avg_in_acts <= mid)
        if cnt <= p*len(avg_in_acts):
            l = mid
        else:
            r = mid
    
    plot(avg_in_acts, np.std(in_acts, axis=0), f'Input Dataset: {input_dataset}')
    
    c = l
    print(f'Input Dataset: {input_dataset} Chosen c for p = {p}: {c}')
    for out_dataset in output_datasets:
        out_acts = np.loadtxt('{base_dir}/{out_ds}/out_acts.txt'.format(base_dir=base_dir, out_ds=out_dataset))
        avg_out_acts = np.mean(out_acts, axis=0)
        cnt = np.sum(avg_out_acts <= c)
        print(f'Dataset: {out_dataset} p: {cnt/len(avg_out_acts)}')
        plot(avg_out_acts, np.std(out_acts, axis=0), f'Output Dataset: {out_dataset}')
    