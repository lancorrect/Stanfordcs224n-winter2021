import torch
from collections import Counter
import numpy as np
'''
data_dir = './data/train.conll'
index = 0
with open(data_dir) as f:
    for line in f.readlines():
        sp = line.strip().split('\t')
        word = sp[1].lower()
        pos = sp[4]
        head = int(sp[6])
        label = sp[7]
        print(sp)
        print(word)
        print(pos)
        print(head)
        print(label)
        break'''

l = np.random.randint(10,size=10)
counter = Counter(l)
print(l)
print(counter.most_common())
print(counter.most_common()[0])
print(counter.most_common()[0][0])