'''
Expected a tensor of logits at the input: C x [N x 5 x 2 x 14]

    List Dim: Class
            [C] # 14 - label

    Dim 0: Data point
            [N] # Number of successful adversarial attacks for class

    Dim 1: Seed
            [5] # 5 models in ensemble

    Dim 2: Original or Adversarial
            [2]

    Dim 3: Emotion Logit predictions
            [14] # 14 classes


Use the Mahalanobis distance in the logit space as a metric for
each original and adv sample.

Calculate the best F1-score for this metric to detect
adversarial samples.
'''

import numpy as np
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from f_score_uncertainty_all import get_best_f_score


def calculate_per_class_dist(vector, class_mean, inv_cov):
    diff = vector - class_mean
    # print(diff)
    half = np.matmul(inv_cov, diff)
    return np.dot(diff, half)

def calculate_mahalanobis(vector, class_means, inv_cov):
    # Select closest class conditional distance
    dists = []
    for class_mean in class_means:
        dists.append(calculate_per_class_dist(vector, class_mean, inv_cov))
    return min(dists)


if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILENAME', type=str, help='.npz file with list of logits array')
    commandLineParser.add_argument('OUT', type=str, help='.png file for pr curve')

    args = commandLineParser.parse_args()
    filename = args.FILENAME
    out_file = args.OUT

    NUM_CLASSES = 14

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/f_score_mahalanobis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the logits array list
    # Keep only seed 1 logits (only model used for detector in logit space)
    logits_dict = np.load(filename)
    train_original_logits_list = []
    test_original_logits_list = []
    test_adv_logits_list = []
    FRAC = 0.8
    for i in range(NUM_CLASSES):
        logits = np.squeeze(logits_dict[f'arr_{i}'][:,0,:,:])
        train_logits = logits[:int(FRAC*len(logits)),:,:]
        test_logits = logits[int(FRAC*len(logits)):,:,:]

        train_original_logits_list.append(np.squeeze(train_logits[:,0,:]))
        test_original_logits_list.append(np.squeeze(test_logits[:,0,:]))
        test_adv_logits_list.append(np.squeeze(test_logits[:,1,:]))

     # Calculate class specific means
     # Calculate an averaged tied covariance matrix
    class_means = []
    cov = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        class_mean = np.mean(train_original_logits_list[i], axis=0)
        class_means.append(class_mean)

        class_cov = np.cov(train_original_logits_list[i], rowvar=False)
        cov += class_cov
    cov = cov/NUM_CLASSES

    # Calculate Mahalanobis distances per test data point
    inv_cov = np.linalg.inv(cov)

    original_dists = []
    for logits in np.concatenate(test_original_logits_list):
        original_dists.append(calculate_mahalanobis(logits, class_means, inv_cov))

    adv_dists = []
    for logits in np.concatenate(test_adv_logits_list):
        adv_dists.append(calculate_mahalanobis(logits, class_means, inv_cov))

    # print(original_dists)
    # print(adv_dists)

    dists = np.asarray(original_dists+adv_dists)
    labels = [0]*len(original_dists) + [1]*len(adv_dists)

    # Calculate best F1-score
    precision, recall, _ = precision_recall_curve(labels, dists)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    # plot all the data
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(f"F1={best_f1:.2f}", (best_recall, best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file)
