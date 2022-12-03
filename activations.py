import torch
import numpy as np

def get_hidden_activations(inputs, model, forward_intermediate):
    with torch.no_grad():
        hidden = forward_intermediate(inputs, model)
    
    return hidden

def compute_act_stats(base_dir, output_datasets, input_dataset, p = 0.9):
    in_acts = np.loadtxt('{base_dir}/in_acts.txt'.format(base_dir=base_dir), delimiter='\n')
    max_in_acts = np.max(in_acts, axis=0)
    l = 0.0
    r = 1e6
    for _ in range(50):
        mid = (l+r)/2.0
        cnt = np.sum(max_in_acts <= mid)
        if cnt <= p*len(max_in_acts):
            l = mid
        else:
            r = mid
    
    c = l
    print(f'Chosen c for p = {p}: {c}')
    for out_dataset in output_datasets:
        out_acts = np.loadtxt('{base_dir}/{out_ds}/out_acts.txt'.format(base_dir=base_dir, out_ds=out_dataset), delimiter='\n')
        max_out_acts = np.max(out_acts, axis=0)
        cnt = np.sum(max_out_acts <= c)
        print(f'Dataset: {out_dataset} p: {cnt/len(max_out_acts)}')
    