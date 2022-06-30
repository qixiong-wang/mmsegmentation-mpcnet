
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
# pickle.load()
import numpy as np
import pdb
import torch


model_dict = torch.load('work_dirs/fpn_segmentor_vaihingen/iter_40000.pth')

query_feature = model_dict['state_dict']['decode_head.cls_emb'][0]

tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
query_feature_tsne = tsne.fit_transform(query_feature)

plt.scatter(query_feature_tsne, query_feature_tsne, marker='o')
        # plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
plt.savefig('query_feature_vis.png')



# CLASSES = ('Passenger-Ship', 'Motorboat', 'Fishing-Boat',
#             'Tugboat', 'other-ship', 'Engineering-Ship', 'Liquid-Cargo-Ship',
#             'Dry-Cargo-Ship', 'Warship', 'Small-Car', 'Bus',
#             'Cargo-Truck', 'Dump-Truck', 'other-vehicle', 'Van',
#             'Trailer', 'Tractor', 'Excavator', 'Truck-Tractor',
#             'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
#             'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',
#             'other-airplane', 'Baseball-Field', 'Basketball-Court',
#             'Football-Field', 'Tennis-Court', 'Roundabout', 'Intersection', 'Bridge')



# markers = ['o','*','^','s','D','v','v','p','h','D']
# colors = ['purple','blue','red','green','yellow','lime','deeppink','orange','cyan','limegreen','black']
# cls_map = {c: i
#             for i, c in enumerate(CLASSES)
#             }

# cls_feature=[[]for i in range(11)]
# cls_label=[[]for i in range(11)]
# with open('features_origin_18epoch.pkl','rb') as f:
#     results = pickle.load(f)
#     for filename, det_result in results:
#         for cls_idx in range(11):
#             if det_result[cls_idx].shape[0]!=0:
#                 for result in det_result[cls_idx]:
#                     if result[-1]>0.5:
#                         if len(cls_feature[cls_idx])<100:
#                             cls_feature[cls_idx].append(result[5:-1])
                        

# for cls_idx in range(11):
#     cls_label[cls_idx]=np.ones(len(cls_feature[cls_idx]))*cls_idx

# cls_label=np.concatenate(cls_label)
# cls_feature = np.concatenate(cls_feature)


# tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
# cls_tsne_feature = tsne.fit_transform(query_feature)
# for cls_idx in range(11):

#     indices = cls_label==cls_idx
#     plt.scatter(cls_tsne_feature[indices, 0],  cls_tsne_feature[indices, 1],s=20,color = colors[cls_idx], marker='o',label = CLASSES[cls_idx])
#         # plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
# plt.savefig('feature_origin18_epoch_vis.png')



        # ann_root='/home/wangqx/FAIR1M/split_ms/train/annfiles/'
        # ann_filename=filename.replace('png','txt')
        # ann_filename = os.path.join(ann_root,ann_filename)
        # with open(ann_filename) as f_ann:
        #     s = f_ann.readlines()
        #     for si in s:
        #         bbox_info = si.split()
        #         poly = np.array(bbox_info[:8], dtype=np.float32)
        #         try:
        #             x, y, w, h, a = poly2obb_np(poly, 'le90')
        #         except:  # noqa: E722
        #             continue
        #         cls_name = bbox_info[8]
        #         difficulty = int(bbox_info[9])
        #         label = cls_map[cls_name]
        #         gt_bboxes = []
        #         gt_labels = []
        #         gt_polygons = []
        #         gt_bboxes.append([x, y, w, h, a])
        #         gt_labels.append(label)
        #         gt_polygons.append(poly)

        # gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        # gt_labels = np.array(gt_labels, dtype=np.float32)
        # gt_polygons = np.array(gt_polygons, dtype=np.float32)

            
        # ious = box_iou_rotated(
        #     torch.from_numpy(det_bboxes).float(),
        #     torch.from_numpy(gt_bboxes).float()).numpy()

# # for each det, the max iou with all gts
# ious_max = ious.max(axis=1)
# # for each det, which gt overlaps most with it
# ious_argmax = ious.argmax(axis=1)
# # sort all dets in descending order by scores
# sort_inds = np.argsort(-det_bboxes[:, -1])



# plt.scatter(class_feature[:, 0], class_feature[:, 1],s=50,color = colors[i], marker= markers[i])
# plt.scatter(output[3][i, 0], output[3][i, 1], color=colors[i], marker= markers[i], s=500,label = LABELS[i])
# plt.legend(fontsize=20,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
# plt.savefig('./results/{}_refine_feature'.format(iter))


