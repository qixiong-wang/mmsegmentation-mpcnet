
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
# pickle.load()
import numpy as np
import pdb
import torch
import mmcv

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, pid_labels, queue_labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labelslut: pytorch LongTensor, with shape N+len(LUT)
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    N = dist_mat.size(0) ### positive sample in batch
    dist_ap = torch.zeros(N).cuda()
    dist_an = torch.zeros(N).cuda()

    for i in range(N):
        label = pid_labels[i]

        dist_ap[i] = torch.max(dist_mat[i][queue_labels==label])
        dist_an[i] = torch.min(dist_mat[i][queue_labels != label])

    return dist_ap, dist_an

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

# query_feature = torch.reshape(query_feature,(-1,128)).cpu().numpy()

batch_queue_label = []
for i in range(query_feature.shape[0]):
    batch_queue_label.extend([i%16]*query_feature.shape[1])

batch_queue_label=torch.tensor(batch_queue_label).cuda()
dist_mat = euclidean_dist(query_feature.reshape(-1,query_feature.shape[-1]), query_feature.reshape(-1,query_feature.shape[-1]))
dist_ap, dist_an = hard_example_mining(dist_mat,batch_queue_label,batch_queue_label)


query_feature = torch.reshape(query_feature,(-1,128)).cpu().numpy()
# tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
# query_feature_tsne = tsne.fit_transform(query_feature)


# query_feature_tsne = manifold.Isomap(n_neighbors=5, n_components=2, n_jobs=-1).fit_transform(query_feature)
# query_feature_tsne = manifold.SpectralEmbedding(n_components=2, n_jobs=-1).fit_transform(query_feature)
query_feature_tsne = manifold.MDS(n_components=2, n_jobs=-1).fit_transform(query_feature)
# query_feature_tsne = manifold.LocallyLinearEmbedding(n_components=2, n_jobs=-1).fit_transform(query_feature)


query_feature_tsne = torch.from_numpy(query_feature_tsne).cuda()

dist_mat_tsne = euclidean_dist(query_feature_tsne.reshape(-1,2), query_feature_tsne.reshape(-1,2))
dist_ap_tsne, dist_an_tsne = hard_example_mining(dist_mat_tsne,batch_queue_label,batch_queue_label)
query_feature_tsne = query_feature_tsne.cpu().numpy()
query_feature_tsne = np.reshape(query_feature_tsne,(64,-1,2))


for i in range(64):
    try:
        color_idx = int(i%16)
        plt.scatter(query_feature_tsne[i,:,0], query_feature_tsne[i,:,1], marker='o',color = colors[color_idx])
    except:
        import pdb
        pdb.set_trace()
    #     plt.scatter(cls_tsne_feature[indices, 0],  cls_tsne_feature[indices, 1],s=20,color = colors[cls_idx], marker='o',label = CLASSES[cls_idx])
    # 
    # plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
plt.savefig('query_feature_vis_20220722.png')


