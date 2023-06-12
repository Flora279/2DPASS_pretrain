import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import tqdm
import yaml

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):

        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    
class SemKITTIRGB(Dataset):
    def __init__(self):
        self.project_scale = 2
        self.output_scale = 1
        self.scans = []    #store all training seqs scans

        # path = '/home/jzhang2297/data/KITTI_Odometry/dataset/sequences'
        path = '/storage/data/zhangjy/SemanticKITTI/kitti/dataset/sequences'
        seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        total_files = []
        for seq in seqs:
            seq_path = os.path.join(path, seq, 'image_2')
            for _,_,files in os.walk(seq_path):
                for file in files:
                    img_path = os.path.join(seq_path, file)
                    total_files.append(img_path)
                    self.scans.append({"img_path": img_path})

        print('all training imgs in SemKITTI', len(self.scans))

    def __len__(self):
        return len(self.scans)


    def __getitem__(self, index):
        data = {}
        scan = self.scans[index]  # individual 2d scan

        img_path = scan["img_path"]
        data["img_path"] = img_path

        return img_path, None

def fast_hist(a, b, n):   #(gt, pred, num_class) gt: 0~12 & 255, pred: 0~12 int
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):         # IoU = TP/(TP+FN+FP)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_prec(hist):       # precision = TP/(TP+FP)
    return np.diag(hist) / hist.sum(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
    sam_checkpoint = '/public/home/zhangjy/weaksup/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    text_prompt = args.text_prompt
    device = args.device
    # load Grounding Dino model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    source_dataset = SemKITTIRGB()
    hist = np.zeros((2,2))
    label_mapping = '/storage/data/zhangjy/SemanticKITTI/kitti/semantic-kitti.yaml'  # map to sem 20 claases
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']
    #np.save('hist_test.npy', hist)
    print('seq 09 or seq 10')
    for batch in tqdm.tqdm(source_dataset):
        image_path, _ = batch  # rgb_node_type_vec = world 0, ground 1
        seq = image_path.split('/')[-3]
        scan = image_path.split('/')[-1].split('.')[0]
        if seq == '09' or seq == '10':
            mapped_points = np.load('/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mapped_pts.npy'.format(seq, scan+'.bin'))
            mask = np.load('/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mask.npy'.format(seq, scan+'.bin'))
            # read gt point cloud labels
            label_path = '/storage/data/zhangjy/SemanticKITTI/kitti/dataset/sequences/{0}/labels/{1}.label'.format(seq, scan)
            labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
            labels = labels & 0xFFFF
            labels_90fov = labels[mask]     # gt labels in 90 fov
            labels_90fov_trainid = np.vectorize(learning_map.__getitem__)(labels_90fov)

            image_pil, image = load_image(image_path)
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size

            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
            if len(transformed_boxes) == 0:
                # if there is no prompt object are found by groundingDino in this img
                total_masks = np.zeros(image_pil.size[::-1])

            else:
                # apply SAM predictor based on the generated bounding box
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(device),
                    multimask_output=False,)
                total_masks = torch.sum(masks.type(torch.int32), dim=0)[0].cpu().numpy()

            # != 0 is the total mask area for this prompt

            pred = np.zeros(total_masks.shape)
            pred[total_masks != 0] = 1          # 0 as unlabeled, 1 as trunk

            h = np.floor(mapped_points[:, 0]).astype(int)
            w = np.floor(mapped_points[:, 1]).astype(int)
            pc_pred = pred[h, w].astype(np.int64)       # prediction in 90 fov point cloud shape
            pc_gt = np.zeros(labels_90fov_trainid.shape)
            pc_gt[labels_90fov_trainid == 16] = 1           # set gt trunk label to 1, others 0

            hist += fast_hist(pc_gt.flatten(), pc_pred.flatten(), 2)            # two classes: unlabeled 0 & trunk 1

    np.save('hist_09_10.npy', hist)
'''
iou = per_class_iu(hist)
print('iou', iou)
precision = per_class_prec(hist)
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
'''


