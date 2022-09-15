
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
# pickle.load()
import numpy as np
import pdb
import torch
import mmcv
from sklearn import decomposition 


query_features= mmcv.load('vaigingen_deeplab.pkl')

for i in range(len(query_features[0])):
    img_name = query_features[0][i][0].split('/')[-1].split('.')[0]
    mmcv.dump([query_features[0][i][1],query_features[1][i]],'feature_vis/{}.pkl'.format(img_name))