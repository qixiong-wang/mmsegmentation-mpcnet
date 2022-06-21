import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)


@torch.no_grad()
def all_gather_tensor(x, gpu=None, save_memory=False):
    
    rank, world_size = get_dist_info()

    if not save_memory:
        # all gather features in parallel
        # cost more GPU memory but less time
        # x = x.cuda(gpu)
        x_gather = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_gather, x, async_op=False)
#         x_gather = torch.cat(x_gather, dim=0)
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        container = torch.empty_like(x).cuda(gpu)
        x_gather = []
        for k in range(world_size):
            container.data.copy_(x)
            print("gathering features from rank no.{}".format(k))
            dist.broadcast(container, k)
            x_gather.append(container.cpu())
#         x_gather = torch.cat(x_gather, dim=0)
        # return cpu tensor
    return x_gather

def undefined_l_gather(features,pid_labels):
    resized_num = 10000
    pos_num = min(features.size(0),resized_num)
    if features.size(0)>resized_num:
        print(f'{features.size(0)}out of {resized_num}')
    resized_features = torch.empty((resized_num,features.size(1))).to(features.device)
    resized_features[:pos_num,:] = features[:pos_num,:]
    resized_pid_labels = torch.empty((resized_num,)).to(pid_labels.device)
    resized_pid_labels[:pos_num] = pid_labels[:pos_num]
    pos_num = torch.tensor([pos_num]).to(features.device)
    all_pos_num = all_gather_tensor(pos_num)
    all_features = all_gather_tensor(resized_features)
    all_pid_labels = all_gather_tensor(resized_pid_labels)
    gather_features = []
    gather_pid_labels = []
    for index,p_num in enumerate(all_pos_num):
        gather_features.append(all_features[index][:p_num,:])
        gather_pid_labels.append(all_pid_labels[index][:p_num])
    gather_features = torch.cat(gather_features,dim=0)
    gather_pid_labels = torch.cat(gather_pid_labels,dim=0)
    return gather_features,gather_pid_labels



class Large_batch_queue_classwise(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_classes=37, number_of_instance=2, feat_len=256):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(Large_batch_queue_classwise, self).__init__()
        self.num_classes = num_classes
        self.register_buffer("large_batch_queue", torch.zeros(num_classes, number_of_instance, feat_len))
        self.register_buffer("tail", torch.zeros(num_classes).long())


    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        # if features.get_device() == 0:
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     dist.barrier()
        
        # gather_features,gather_pid_labels = undefined_l_gather(features,pid_labels)
        
        with torch.no_grad():
            for indx, label in enumerate(pid_labels):
                label=int(label)
                if label >= 0 and label<self.num_classes:

                    self.large_batch_queue[label,self.tail[label]] = features[indx]
                    
                    # self.large_batch_queue[label,self.tail[label]] = torch.mean(features[pid_labels==label],dim=0)
                    self.tail[label]+=1
                    if self.tail[label] >= self.large_batch_queue.shape[1]:
                        self.tail[label] -= self.large_batch_queue.shape[1]

        return self.large_batch_queue