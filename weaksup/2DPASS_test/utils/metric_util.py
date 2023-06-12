import numpy as np
import torch

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist

def fast_hist_torch(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = torch.bincount(
        n * label[k] + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def fast_hist_crop_torch(output, target, unique_label):
    hist = fast_hist_torch(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist