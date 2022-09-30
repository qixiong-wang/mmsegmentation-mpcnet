from cmath import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Nce_contrast_loss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, num_classes=6):
        super(Nce_contrast_loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_classes = num_classes

    def forward(self, pid_features,pid_labels, large_batch_queue):
        """
        Does not calculate noise inputs with label -1
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        #print(inputs.shape, targets.shape)

        avai_labels =[]
        avai_features =[]
        for indx, label in enumerate(pid_labels):
            # if label >= 0 and label<self.num_classes:
            label=label%self.num_classes
            avai_labels.append(label)
            avai_features.append(pid_features[indx])

        avai_labels=torch.stack(avai_labels).cuda()
        avai_labels_onehot = F.one_hot(avai_labels,self.num_classes).float()
        queue_labels_onehot =  F.one_hot(torch.arange(self.num_classes).unsqueeze(1).expand(self.num_classes,100).repeat(3,1).reshape(-1)).float().cuda()

        same_label_matrix = torch.matmul(avai_labels_onehot,queue_labels_onehot.permute(1,0))

        resized_queue = torch.permute(large_batch_queue.reshape(-1,large_batch_queue.shape[-1]),(1,0))
        resized_queue = F.normalize(resized_queue,dim=0)
        avai_features=torch.stack(avai_features).cuda()
        avai_features = F.normalize(avai_features,dim=1)

        logits=torch.matmul(avai_features,resized_queue) 
        # dist_label = same_label_matrix*2-1
        # logits_ranking = torch.multiply(logits,dist_label)
        prob =torch.sum(torch.multiply(same_label_matrix,F.softmax(logits,dim=1)),dim=1)

        loss = torch.sum(-torch.log(prob))/prob.shape[0]
        # torch.cuda.empty_cache()
        return loss
