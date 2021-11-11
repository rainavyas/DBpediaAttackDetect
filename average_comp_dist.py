'''
Use training data to define chosen embedding space eigenvectors
For original and attacked test data determine the average size of the components in the eigenvector directions
Plot this against eigenvalue rank
'''

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tools import get_default_device
import matplotlib.pyplot as plt
from models import ElectraSequenceClassifier
from transformers import ElectraTokenizer
from layer_handler import Electra_Layer_Handler
from linear_classifier import batched_get_layer_embedding, load_test_adapted_data_sentences
from data_prep import DataTensorLoader
from pca_tools import get_covariance_matrix, get_e_v

def get_embeddings_batched(sen_list, handler, tokenizer, device):

    encoded_inputs = tokenizer(sen_list, padding='max_length', truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    print(ids.size())

    ds = TensorDataset(ids, mask)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    collected = []
    for count, (i, m) in enumerate(dl):
        print(count)
        embeddings = handler.get_layern_outputs(i, m, device)
        collected.append(embeddings)
    return torch.cat(collected)

def stds(original, attack):
    '''
    original: Tensor [num_ranks x num_data_points]
    attack: Tensor [num_ranks x num_data_points]

    Return:
        The average (across rank) of the number of
        standard deviations between original and
        attack at each rank position

    This gives a measure of how out of distribution
    the attack is from the original
    '''
    with torch.no_grad():
        original_mean = torch.mean(original, dim=1)
        attack_mean = torch.mean(attack, dim=1)
        original_std = torch.std(original, dim=1)
        diff = torch.abs(attack_mean - original_mean)
        std_diff = diff/original_std
        return std_diff

def get_all_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector (rank), calculates the
    magnitude of components in that direction for each
    data point

    Returns:
        Tensor [num_ranks, num_data_points]
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.abs(torch.einsum('bi,ji->bj', X, eigenvectors))
    return torch.transpose(comps, 0, 1)

def get_avg_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector, calculates average (across batch)
    magnitude of components in that direction
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.einsum('bi,ji->bj', X, eigenvectors)
        # Get average of magnitude for each eigenvector rank
        avg_comps = torch.mean(torch.abs(comps), dim=0)
    return avg_comps


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DIR', type=str, help='attacked data base directory')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")
    commandLineParser.add_argument('--N', type=int, default=6, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    commandLineParser.add_argument('--num_points_train', type=int, default=800, help="train points to caluclate PCA projection")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number of data points to use test")
    commandLineParser.add_argument('--error_layer_num', type=int, default=0, help="layer to calculate error for")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    out_file = args.OUT_FILE
    layer_num = args.layer_num
    N = args.N
    cpu_use = args.cpu
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    error_layer_num = args.error_layer_num

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/average_comp_dist.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the Sentiment Classifier model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Electra_Layer_Handler(model, layer_num=layer_num)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Use training data to get eigenvector basis for CLS token at correct layer
    # Load the data as tensors
    dataloader = DataTensorLoader('electra')
    input_ids, mask, _ = dataloader.get_train()
    indices = torch.randperm(len(input_ids))[:num_points_train]
    input_ids = input_ids[indices]
    mask = mask[indices]
    with torch.no_grad():
        layer_embeddings = handler.get_layern_outputs(input_ids, mask)
        CLS_embeddings = layer_embeddings[:,0,:].squeeze()
        correction_mean = torch.mean(CLS_embeddings, dim=0)
        cov = get_covariance_matrix(CLS_embeddings)
        e, v = get_e_v(cov)

    # Load the test data
    original_list, attack_list = load_test_adapted_data_sentences(base_dir, num_points_test)
    print("Loaded data")

    # Get embeddings
    original_embeddings = batched_get_layer_embedding(original_list, handler, tokenizer, device)
    attack_embeddings = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

    # Get average components against rank
    original_avg_comps = get_avg_comps(original_embeddings, v, correction_mean)
    attack_avg_comps = get_avg_comps(attack_embeddings, v, correction_mean)

    # Plot the results
    ranks = np.arange(len(original_avg_comps))
    plt.plot(ranks, original_avg_comps, label='Original')
    plt.plot(ranks, attack_avg_comps, label='Attacked')
    plt.yscale('log')
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Average Component Size')
    plt.legend()
    plt.savefig(out_file)
    plt.clf()

    # Report std diff between attack and original curves
    original_comps = get_all_comps(original_embeddings, v, correction_mean)
    attack_comps = get_all_comps(attack_embeddings, v, correction_mean)

    std_diffs = stds(original_comps, attack_comps)
    print("OOD metric", torch.mean(std_diffs))

    # # Plot std_diffs ranked by size
    # std_diffs_ordered, _ = torch.sort(std_diffs)
    # ranks = np.arange(len(std_diffs_ordered))
    
    # plt.plot(ranks, std_diffs_ordered)
    # plt.xlabel('std difference rank')
    # plt.ylabel('std difference')
    # plt.savefig('std_'+out_file)
    # plt.clf()

    # Determine normalised error sizes in input embedding layer
    #     - l2
    #     - l-inf

    handler = Electra_Layer_Handler(model, layer_num=error_layer_num)

    # Get all layer embeddings
    original_embeddings = get_embeddings_batched(original_list, handler, tokenizer, device)
    attack_embeddings = get_embeddings_batched(attack_list, handler, tokenizer, device)

    orig = torch.reshape(torch.abs(original_embeddings), (len(original_list), -1))
    l2s_orig_avg = torch.mean(torch.sqrt(torch.sum(orig**2, dim=1)))
    linfs_orig, _ = torch.max(orig, dim=1)
    linfs_orig_avg = torch.mean(linfs_orig)

    diffs = torch.reshape(torch.abs(attack_embeddings-original_embeddings), (len(attack_list), -1))

    l2s = torch.sqrt(torch.sum(diffs**2, dim=1))/l2s_orig_avg
    print(f'l2: mean={torch.mean(l2s)} std={torch.std(l2s)}')

    linfs, _ = torch.max(diffs, dim=1)
    linfs = linfs/linfs_orig_avg
    print(f'l-inf: mean={torch.mean(linfs)} std={torch.std(linfs)}')
