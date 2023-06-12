import os
import yaml
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from pathlib import Path
from nuscenes.utils import splits

REGISTERED_PC_DATASET_CLASSES = {}


city_info = {
    'label2semkitti':
    [[0, 9],      # road
     [1, 11],      # sidewalk
     [2, 13],      # building
     [3, 0],    # wall
     [4, 14],      # fence
     [5, 18],     # pole
     [6, 0],    # traffic light
     [7, 19],     # traffic sign
     [8, 15],      # vegetation
     [9, 17],     # terrain,
     [10, 0],   # sky
     [11, 6],     # person
     [12, 0],   # rider -> bicyclist, weak lbl
     [13, 1],     # car
     [14, 4],     # truck
     [15, 5],   # bus    -> other-vehicle
     [16, 5],   # train  -> other-vehicle
     [17, 3],     # motorcycle
     [18, 2],     # bicycle
    ]
}
city_mapping = np.array(city_info['label2semkitti'], dtype=np.int64)
city_map_vector = np.zeros((city_mapping.shape[0],), dtype=np.int64)
for original_label, train_label in city_mapping:
    city_map_vector[original_label] = train_label

def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


@register_dataset
class SemanticKITTI(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)

        # if imageset == 'val':                
        #     self.im_idx = self.im_idx[:100] 
        # else:
        #     self.im_idx = self.im_idx[:100] 
            
        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)


    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']


        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len

        return data_dict, self.im_idx[index]


@register_dataset
class SemanticKITTI_90fov_plabel(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        print('enter SemanticKITTI_90fov_plabel')
        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)

        # if imageset == 'val':
        #     self.im_idx = self.im_idx[:100]
        # else:
        #     self.im_idx = self.im_idx[:100]

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        seq = self.im_idx[index].split('/')[-3]
        scan = self.im_idx[index].split('/')[-1]  # e.g. 000000.bin
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        path = '/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mapped_pts.npy'.format(seq, scan)
        mask_path = '/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mask.npy'.format(seq, scan)
        map_pts = np.load(path)  # 90 fov 3d points in 2d coordinates, around 2w num per scan
        mask = np.load(mask_path)  # indicate which points are projected to image 90 fov
        labels = np.zeros(len(annotated_data))
        # use floor operation to get projected 2d coordinates
        h = np.floor(map_pts[:, 0]).astype(int)
        w = np.floor(map_pts[:, 1]).astype(int)
        # get InternImage prediction of common (2d-3d) img pixels
        intern_pred_path = '/storage/data/zhangjy/SemanticKITTI/kitti/dataset/sequences/{0}/image_2_pred_new/{1}.png'.format(
            seq, scan.split('.')[0])
        intern_pred = cv2.imread(intern_pred_path, cv2.IMREAD_GRAYSCALE)  # get prediction in cityscapes label id: 0~18
        common_pixels_pred = city_map_vector[
            intern_pred[h, w].astype(np.int64, copy=False)]  # map city_pred to semkitti id for training.

        # 把其他270度点以及筛掉的90度点label=0
        labels[mask] = common_pixels_pred
        # shape (123433, 1), 0~19

        data_dict = {}
        if seq != '08':
            data_dict['labels'] = labels[:, np.newaxis].astype(np.uint8)
        if seq == '08':
            data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['xyz'] = points
        # data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len

        return data_dict, self.im_idx[index]

@register_dataset
class SemanticKITTI_90fov_plabel_high_conf(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        print('enter SemanticKITTI_90fov_plabel_high_conf')
        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)

        # if imageset == 'val':
        #     self.im_idx = self.im_idx[:100]
        # else:
        #     self.im_idx = self.im_idx[:100]

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        seq = self.im_idx[index].split('/')[-3]
        scan = self.im_idx[index].split('/')[-1]  # e.g. 000000.bin
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        path = '/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mapped_pts.npy'.format(seq, scan)
        mask_path = '/public/home/zhangjy/weaksup/mapped_points/{0}_{1}_mask.npy'.format(seq, scan)
        map_pts = np.load(path)  # 90 fov 3d points in 2d coordinates, around 2w num per scan
        mask = np.load(mask_path)  # indicate which points are projected to image 90 fov
        labels = np.zeros(len(annotated_data))
        # use floor operation to get projected 2d coordinates
        h = np.floor(map_pts[:, 0]).astype(int)
        w = np.floor(map_pts[:, 1]).astype(int)
        # get InternImage prediction of common (2d-3d) img pixels
        intern_pred_path = '/storage/data/zhangjy/SemanticKITTI/kitti/dataset/sequences/{0}/image_2_pred_new/{1}.png'.format(
            seq, scan.split('.')[0])
        intern_pred = cv2.imread(intern_pred_path, cv2.IMREAD_GRAYSCALE)  # get prediction in cityscapes label id: 0~18
        confidence_path = '/storage/data/zhangjy/SemanticKITTI/kitti/dataset/sequences/{0}/image_2_pred_new_maxprob/{1}.png'.format(
            seq, scan.split('.')[0])
        confidence = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
        common_pixels_pred = city_map_vector[
            intern_pred[h, w].astype(np.int64, copy=False)]  # map city_pred to semkitti id for training.
        conf_90fov = confidence[h, w].astype(np.int64)
        # only train with hgih confident plabel
        common_pixels_pred[conf_90fov <= 3] = 0

        # 把其他270度点以及筛掉的90度点label=0
        labels[mask] = common_pixels_pred
        # shape (123433, 1), 0~19

        data_dict = {}
        if seq != '08':
            data_dict['labels'] = labels[:, np.newaxis].astype(np.uint8)
        if seq == '08':
            data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['xyz'] = points
        # data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len

        return data_dict, self.im_idx[index]

@register_dataset
class nuScenes(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                if imageset == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']

        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])
                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'cam_token': cam_token}
                    )

    def __getitem__(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor',
                                        pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor',
                                      cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img'] = image
        data_dict['calib_infos'] = calib_infos
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)

        return data_dict, lidar_sample_token


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name
