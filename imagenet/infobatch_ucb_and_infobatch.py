import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union

class InfoBatch(Dataset):
    #this is a UCB+infobatch implementation using InfoBatch framework
    def __init__(self, dataset, ratio = 0.5, momentum = 0.8, batch_size = None, num_epoch=None, delta = None):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.ema = np.full(len(self.dataset),0.)
        self.varema = np.full(len(self.dataset),0.)
        self.transform = dataset.transform
        self.weights = np.full(len(self.dataset),1.)
        self.save_num = 0
        self.momentum = momentum
        self.batch_size = batch_size
        self.total_time = 0
        self.seq = np.array([])
        self.counter = 0

    def __setscore__(self, indices, values):
        vars = (values - self.ema[indices])**2
        self.ema[indices] = np.where(self.ema[indices]>0, (1-self.momentum)*self.ema[indices] + self.momentum*values,values)
        self.varema[indices] = (1-self.momentum)*self.varema[indices] + self.momentum*vars

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight

    def prune(self, leq = False):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        start = time.time()
        scores = self.ema + self.varema
        if self.counter%10==1:
            samples = list(range(len(self.dataset)))
            ordered = sorted(samples,key=lambda i: scores[i])
            pruned_samples = np.array(ordered[int(0.3*len(self.dataset)):])
            self.seq = pruned_samples
        else:
            pruned_samples = self.seq if len(self.seq)>0 else np.array(list(range(len(self.dataset))))

        scores_pruned = scores[pruned_samples]
        b = scores<scores_pruned.mean()
        well_learned_samples = np.intersect1d(np.where(b)[0],pruned_samples)
        pruned_samples = np.intersect1d(np.where(~b)[0],pruned_samples)
        selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)))
        self.reset_weights()
        if len(selected)>0:
            self.weights[selected]=1/self.ratio
            pruned_samples = np.concatenate((pruned_samples,selected))

        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        self.total_time+=time.time()-start
        self.counter+=1
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1, cycle=10):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.cycle = cycle
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        elif (self.seed % self.cycle)==2:
            self.seq = self.infobatch_dataset.prune(self.seed>1)
        elif self.seq is None:
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune(self.seed>1)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas

        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        indexes_of_indexes = indices
#         indexes_of_indexes = super().__iter__()  # change this line
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
#         return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

def is_master():
    if not torch.distributed.is_available():
        return True

    if not torch.distributed.is_initialized():
        return True

    if torch.distributed.get_rank()==0:
        return True

    return False

def split_index(t):
    low_mask = 0b111111111111111
    low = torch.tensor([x&low_mask for x in t])
    high = torch.tensor([(x>>15)&low_mask for x in t])
    return low,high

def recombine_index(low,high):
    original_tensor = torch.tensor([(high[i]<<15)+low[i] for i in range(len(low))])
    return original_tensor