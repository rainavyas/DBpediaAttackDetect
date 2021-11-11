'''
Generate file to store all original and attacked
predicted logits for all successful adversarial attacks

Output .npz file stores information as list of numpy arrays: C x [N x 5 x 2 x 14]

    List Dim: Class
            [C] # 14 - label of class

    Dim 0: Data point
            [N] # Number of successful adversarial attacks for class

    Dim 1: Seed
            [5] # 5 models in ensemble

    Dim 2: Original or Adversarial
            [2]

    Dim 3: Emotion Logit predictions
            [14] # 14 classes
'''

import json
import argparse
import os
import sys
from tools import get_default_device
import numpy as np
import torch
from models import ElectraSequenceClassifier
from transformers import ElectraTokenizer

def load_data(base_dir):
    '''
    Returns data as list from attacked data directory
    '''

    data_list = []
    for i in range(20000):
        fname = base_dir + '/'+str(i)+'.txt'
        try:
            with open(fname, 'r') as f:
                item = json.load(f)
            data_list.append(item)
        except:
            print("Failed to load", i)
    return data_list

def keep_only_success(data_list):
    '''
    data_list: list
        [item1, item2, ...]
        where,
            item: dict
                with keys:
                    sentence
                    updated sentence
                    original prob
                    updated prob
                    true label
    Filter list to keep only items where adv attack worked
    '''
    new_data_list = []
    for item in data_list:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        label = item['true label']

        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))

        if original_pred != label:
            continue

        if updated_pred != original_pred:
            new_data_list.append(item)

    return new_data_list


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='data base directory with original and attack results')
    commandLineParser.add_argument('MODEL_BASE', type=str, help='directory and base model name, excluding seed and extension')
    commandLineParser.add_argument('OUT', type=str, help='output .npz file to save list of logits array to')
    commandLineParser.add_argument('--num_models', type=int, default=5, help="Specify number of models in ensemble")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    base_dir = args.DIR
    model_base = args.MODEL_BASE
    out_file = args.OUT
    num_models = args.num_models
    cpu_use = args.cpu

    NUM_CLASSES = 14

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_ensemble_logits_by_class.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load all the data from attacked data directory
    data_list = load_data(base_dir)

    # Keep only successful adversarial attack data points
    data_list = keep_only_success(data_list)

    # Initialise arrays to store all logit information by class
    logits_list = []
    for i in range(NUM_CLASSES):
        logits_list.append(np.zeros((len(data_list), num_models, 2, NUM_CLASSES)))

    # Load all the trained models
    models = []
    for i in range(1, num_models+1):
        model_path = f'{model_base}{i}.th'
        model = ElectraSequenceClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    # Prepare the tokenizer
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Populate the logit information array with model predictions
    class_lengths = [0]*NUM_CLASSES
    for n, item in enumerate(data_list):
        print(f'On {n} of {len(data_list)}')
        original_phrase = item['sentence']
        attack_phrase = item['updated sentence']
        class_label = int(item['true label'])
        phrases = [original_phrase, attack_phrase]

        # Represent as torch tensors
        encoded_inputs = tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
        ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']

        # Get logit predictions
        for m, model in enumerate(models):
            with torch.no_grad():
                preds = model(ids, mask).cpu().detach().numpy()
            logits_list[class_label][n,m] = preds
            class_lengths[class_label] += 1

    # Keep only relevant part of each tensor
    for ind in range(NUM_CLASSES):
        logits_list[ind] = logits_list[ind][:class_lengths[ind]]

    # Save the logit information array
    np.savez(out_file, logits_list[0], logits_list[1], logits_list[2], logits_list[3], logits_list[4], logits_list[5], logits_list[6], logits_list[7], logits_list[8], logits_list[9], logits_list[10], logits_list[11], logits_list[12], logits_list[13])
