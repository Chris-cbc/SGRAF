"""Data provider"""

import torch
import torch.utils.data as data
import jieba
import os
import numpy as np


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'
        # load the raw captions
        self.captions = []
        self.captions = [line.strip() for line in open(loc + '%s_caps.txt' % data_split, 'rb') if line.strip()][
                        :opt.data_size]
        # load the image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)[:opt.data_size]
        # self.images = np.memmap(loc + '%s_ims.npy' % data_split, dtype='float32', shape=(51,280,280))
        self.length = len(self.captions)
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev':
        #     self.length = 5000

    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        caption = self.captions[index]
        vocab = self.vocab
        # convert caption (string) to word ids.
        tokens = [w for w in jieba.cut(caption.lower()) if w not in {" ", "."}]
        caption = [vocab('<start>')]
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, index

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, opt)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    # train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
    #                                   batch_size, True, workers)
    # get the val_loader
    df_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                   batch_size, False, workers)
    return df_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader