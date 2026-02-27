import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


Q_FRAC_BITS = 20
Q_SCALE = 1 << Q_FRAC_BITS
INT32_MIN = -2**31
INT32_MAX =  2**31 - 1

def float_to_q12_20(fp32_array: np.ndarray) -> np.ndarray:
    """
    Convert FP32 numpy array to signed Q12.20 (int32).
    """
    # Ensure FP32 input
    x = fp32_array.astype(np.float32)

    # Scale
    q = np.round(x * Q_SCALE).astype(np.int64)

    # Saturate to int32
    q = np.clip(q, INT32_MIN, INT32_MAX)

    return q.astype(np.int32)

import numpy as np
import os

def repack_weights_for_parallel_filters(
    input_bin_path,
    output_bin_path,
    output_coe_path,
    OUT_C,
    K,
    IN_C,
    P_FILTER,
    little_endian=True
):
    """
    Repack weight binary file for P_FILTER-parallel BMG.

    INPUT:
        int32 (Q-format assumed)
        layout: (OUT_C, WIN_ELEMS)
        filter-major

    OUTPUT MEMORY LAYOUT (depth x P_FILTER):

        for fg in range(F_GROUPS):
            for k in range(WIN_ELEMS):
                addr = fg*WIN_ELEMS + k
                lane0 = filter fg*P_FILTER + 0
                lane1 = filter fg*P_FILTER + 1
                ...
                lane(P_FILTER-1)

    Produces:
        1) Flat int32 stream (for AXI write)
        2) COE file (wide-word memory init)
    """

    # ------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------
    WIN_ELEMS = K * K * IN_C

    if OUT_C % P_FILTER != 0:
        raise ValueError(
            f"OUT_C ({OUT_C}) must be divisible by P_FILTER ({P_FILTER})"
        )

    F_GROUPS = OUT_C // P_FILTER
    DEPTH    = F_GROUPS * WIN_ELEMS

    # ------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------
    weights = np.fromfile(input_bin_path, dtype=np.int32)
    bn = False

    expected = OUT_C * WIN_ELEMS
    if "BINARY.BIN" in input_bin_path:
        expected = expected / 32
        
    if weights.size == expected and "BN" not in input_bin_path:
        weights = weights.reshape(OUT_C, WIN_ELEMS)
        bn = False
    elif weights.size == OUT_C * 2 and "BN" in input_bin_path:
        bn = True
    else:
        raise ValueError(
            f"Expected {expected} int32 weights, got {weights.size}"
        )



    print(f"Loaded weights shape: {weights.shape}")
    print(f"WIN_ELEMS: {WIN_ELEMS}")
    print(f"F_GROUPS:  {F_GROUPS}")
    print(f"DEPTH:     {DEPTH}")

    # ------------------------------------------------------------
    # Pack for parallel filters
    # ------------------------------------------------------------
    packed = np.zeros((DEPTH, P_FILTER), dtype=np.int32)

    for fg in range(F_GROUPS):
        base_filter = fg * P_FILTER
        if bn:
            for lane in range(P_FILTER):
                packed[fg, lane] = weights[base_filter + lane]
        else:
            for k in range(WIN_ELEMS):
                addr = fg * WIN_ELEMS + k

                for lane in range(P_FILTER):
                    packed[addr, lane] = weights[base_filter + lane, k]

    # ------------------------------------------------------------
    # Write flat 32-bit stream
    # ------------------------------------------------------------
    packed_flat = packed.reshape(-1)  # depth * P_FILTER words
    packed_flat.tofile(output_bin_path)

    print("\nBinary packing complete.")
    print(f"Packed shape: {packed.shape}")
    print(f"Total int32 written: {packed_flat.size}")
    print(f"Total bytes: {packed_flat.size * 4}")

    # ------------------------------------------------------------
    # Write COE file
    # ------------------------------------------------------------
    print("Creating COE file...")

    with open(output_coe_path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")

        for addr in range(DEPTH):
            word_hex = lanes_to_hex_word(
                packed[addr],
                little_endian=little_endian
            )

            if addr != DEPTH - 1:
                f.write(word_hex + ",\n")
            else:
                f.write(word_hex + ";\n")

    print("COE file written successfully.")
    print(f"Depth entries: {DEPTH}")
    print(f"Each entry: {P_FILTER} x 32-bit = {P_FILTER*32} bits")
        
def lanes_to_hex_word(lanes, little_endian=True):
    """
    Convert list/array of int32 lanes into one wide hex word.

    lanes[0] becomes LSB if little_endian=True
    """
    word = 0
    P = len(lanes)

    if little_endian:
        for i in range(P):
            word |= (np.uint32(lanes[i]).item() & 0xFFFFFFFF) << (32 * i)
    else:
        for i in range(P):
            word |= (np.uint32(lanes[i]).item() & 0xFFFFFFFF) << (32 * (P - 1 - i))

    width_bits = 32 * P
    width_hex  = width_bits // 4

    return f"{word:0{width_hex}X}"
