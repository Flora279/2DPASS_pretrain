import numpy as np

def fast_hist(a, b, n):   #(gt, pred, num_class) gt: 0~12 & 255, pred: 0~12 int
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):         # IoU = TP/(TP+FN+FP)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_prec(hist):       # precision = TP/(TP+FP)
    return np.diag(hist) / hist.sum(0)

path1 = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/hist_00_01.npy'
path2 = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/hist_02_03.npy'
path3 = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/hist_04_05_06.npy'
path4 = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/hist_07_08.npy'
path5 = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/hist_09_10.npy'

hist1 = np.load(path1)
hist2 = np.load(path2)
hist3 = np.load(path3)
hist4 = np.load(path4)
hist5 = np.load(path5)

hist_total = hist1 + hist2 + hist3 + hist4 + hist5

iou = per_class_iu(hist_total)
print('iou', iou)
precision = per_class_prec(hist_total)
print('precision', precision)
file = open(
    '/public/home/zhangjy/weaksup/plabel_90fov_result_sam_trunk.txt', "a")
class_name = ['unlabeled', 'trunk']
file.write('IoU result:')
file.write('\n')
for classs, class_iou in zip(class_name, iou):
    print('%s : %.2f%%' % (classs, class_iou * 100))
    file.write('%s : %.2f%%' % (classs, class_iou * 100))
    file.write('\n')

file.write('Precision result:')
file.write('\n')
for classs, class_prec in zip(class_name, precision):
    print('%s : %.2f%%' % (classs, class_prec * 100))
    file.write('%s : %.2f%%' % (classs, class_prec * 100))
    file.write('\n')
file.close()