
import pickle
import json
import sys
sys.path.append('../')

from data_loader import DataLoader

with open(r'GraftNet/manually_created_files/read_test_try1_check', 'rb') as f1:
    data = pickle.load(f1)

print(type(data))
