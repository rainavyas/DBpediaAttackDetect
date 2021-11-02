'''
To run jobs on the cpu stack, <=python3.6 is required for the Transformer package
To use the datasets package, >=python3.7 is required
Hence, solution:
1) Use python3.7 to load and save test data
2) Use python3.6 for Transformer only cpu machines, manually loading data without the datasets package
'''

import pickle
from datasets import load_dataset

dataset = load_dataset('dbpedia_14')
data = dataset['test']
texts = data['content']
labels = data['label']

out_path_base = 'data/test'

with open(f'{out_path_base}/texts.pkl', 'wb') as f:
    pickle.dump(texts, f)
with open(f'{out_path_base}/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)