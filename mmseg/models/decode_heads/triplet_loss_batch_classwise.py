import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class TripletLossbatch_classwise(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, num_classes=37):
        super(TripletLossbatch_classwise, self).__init__()
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
        for indx, label in enumerate(torch.unique(pid_labels)):
            if label >= 0 and label<self.num_classes:
                avai_labels.append(label)
                avai_features.append(torch.mean(pid_features[pid_labels==label],dim=0))

        avai_labels=torch.stack(avai_labels).cuda()
        avai_features=torch.stack(avai_features).cuda()
        batch_queue_label=[]

        for i in range(large_batch_queue.shape[0]):
            batch_queue_label.extend([i]*large_batch_queue.shape[1])
        batch_queue_label=torch.tensor(batch_queue_label).cuda()
        dist_mat = euclidean_dist(avai_features, large_batch_queue.reshape(-1,large_batch_queue.shape[-1]))
        dist_ap, dist_an = hard_example_mining(
            dist_mat,avai_labels,batch_queue_label)
        y = dist_an.new().resize_as_(dist_an).fill_(1)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        # torch.cuda.empty_cache()
        return loss
