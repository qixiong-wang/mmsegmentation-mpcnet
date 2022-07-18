
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
# pickle.load()
import numpy as np
import pdb
import torch
import mmcv


model_dict = torch.load('work_dirs/fpn_twins_cascade_dpet_multi_memory_isaid/iter_160000.pth')

query_feature = model_dict['state_dict']['decode_head.large_batch_queue.large_batch_queue']

# tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
# query_feature_tsne = tsne.fit_transform(query_feature)

# plt.scatter(query_feature_tsne, query_feature_tsne, marker='o')
#         # plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
# plt.savefig('query_feature_vis.png')



# from sklearn import manifold
# import matplotlib.pyplot as plt
# import numpy as np
# import pdb
# import torch

# query_features= mmcv.load('vis_feature.pkl')
# temp = []

# colors = ['purple','blue','red','green','yellow','lime','deeppink','orange','cyan','limegreen','black']
colors = ['purple','blue','red','green','yellow','lime','deeppink','orange','cyan','limegreen','black', 'grey', 'navy', 'coral', 'indianred', 'deepskyblue']

# color_1 = ['forestgreen', 'limegreen', 'green', 'lime']
# color_2 = ['darkcyan', 'cyan', 'darkturquoise', 'deepskyblue','cornflowerblue']
# color_3 = ['blue', 'darkblue', 'navy', 'mediumblue']
# color_4 = ['yellow', 'gold', 'darkorange', 'orange']
# color_5 = ['red', 'orangered', 'coral', 'indianred']
# color_6 = ['black', 'dimgray', 'grey', 'darkgray'] 
# colors = []
# for i in range(1,7):
#     colors.extend(eval('color_{}'.format(i)))

# model_dict = torch.load('work_dirs/fpn_segmentor_vaihingen/iter_40000.pth')
# query_feature = model_dict['state_dict']['decode_head.cls_emb'][0]
# for i in query_features[0:2]:
#     temp.extend(i)

# query_feature = torch.cat(temp)

# query_feature = query_feature[:,0].cpu().numpy()

query_feature = torch.reshape(query_feature,(-1,128)).cpu().numpy()

tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
query_feature_tsne = tsne.fit_transform(query_feature)
query_feature_tsne = np.reshape(query_feature_tsne,(64,-1,2))


for i in range(64):
    try:
        color_idx = int(i/4)
        plt.scatter(query_feature_tsne[i,:,0], query_feature_tsne[i,:,1], marker='o',color = colors[color_idx])
    except:
        import pdb
        pdb.set_trace()
    #     plt.scatter(cls_tsne_feature[indices, 0],  cls_tsne_feature[indices, 1],s=20,color = colors[cls_idx], marker='o',label = CLASSES[cls_idx])
    # 
    # plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
plt.savefig('query_feature_vis.png')


