'''
Identical to f_score_uncertainty, but generalized for many
different classification uncertainty measures:

- Expected Pair-Wise KL Divergence (epkl)
- entropy of expected
- expected entropy
- mutual information
- reverse mutual information
- negative confidence

Expected a tensor of logits at the input: [N x 5 x 2 x 14]

    Dim 0: Data point
            [N] # Number of successful adversarial attacks

    Dim 1: Seed
            [5] # 5 models in ensemble

    Dim 2: Original or Adversarial
            [2]

    Dim 3: Emotion Logit predictions
            [14] # 14 classes
'''

import sys
import os
import argparse
import numpy as np
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from uncertainty import ensemble_uncertainties_classification

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':
	# Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILENAME', type=str, help='.npy file with logits array')
    commandLineParser.add_argument('OUT_DIR', type=str, help='directory to save .png files for pr curves to')

    args = commandLineParser.parse_args()
    filename = args.FILENAME
    out_dir = args.OUT_DIR

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/f_score_uncertainty_all.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the logits array
    logits = np.load(filename)
    probs = softmax(logits, axis=-1)
    print("Array shape", probs.shape)

    original_probs = probs[:,:,0,:].squeeze()
    adv_probs = probs[:,:,1,:].squeeze()

    # Transpose to match uncertainty function format
    original_probsT = np.transpose(original_probs, (1, 0, 2))
    adv_probsT = np.transpose(adv_probs, (1, 0, 2))

    # Get uncertainty measure for each data point
    original_uncertainties = ensemble_uncertainties_classification(original_probsT)
    adv_uncertainties = ensemble_uncertainties_classification(adv_probsT)

    # For each uncertainty measure get separate PR curve
    print(len(original_probs), len(adv_probs))
    labels = [0]*len(original_probs) + [1]*len(adv_probs)

    for measure in original_uncertainties.keys():
        original = original_uncertainties[measure]
        adv = adv_uncertainties[measure]
        together = np.concatenate((original, adv))

        # Calculate best F1-score
        precision, recall, _ = precision_recall_curve(labels, together)
        best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

        # plot all the data
        out_file = f'{out_dir}/pr_{measure}.png'
        plt.plot(recall, precision, 'r-')
        plt.plot(best_recall,best_precision,'bo')
        plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(out_file)
        plt.clf()

    # Plot all uncertainty measure PR curves on same graph
    out_file = f'{out_dir}/pr_all_uncertainty.png'
    for measure in original_uncertainties.keys():
        original = original_uncertainties[measure]
        adv = adv_uncertainties[measure]
        together = np.concatenate((original, adv))

        # Calculate best F1-score
        precision, recall, _ = precision_recall_curve(labels, together)
        best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
        plt.plot(recall, precision, label=f'{measure}, F1={best_f1:.2f}')
        plt.plot(best_recall,best_precision,'ko')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(out_file)
