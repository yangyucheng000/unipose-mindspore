import os

import scipy.io
import numpy as np
import glob
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size, init

import scipy.misc
from PIL import Image
import cv2
import src.transforms as transforms
from src.config import config as cfg

def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    image_list = glob.glob(root_dir+'/*.jpg')
    print("image_list len:  ",len(image_list))
    for idx in range(len(image_list)):
        image_list[idx] = image_list[idx].replace('\\', '/')
    return image_list

def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'joints_train' or 'joints_test'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    if mode == 'joints_train':
        # lspnet (14,3,10000)
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints_train.mat').replace('\\', '/'))['joints_train']
        # lms = mat_arr.transpose([2, 1, 0])
        # kpts = mat_arr.transpose([2, 0, 1]).tolist()
    elif mode =='joints_test':
        # lsp (3,14,2000)
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints_test.mat').replace('\\', '/'))['joints_test']
    mat_arr[2] = np.logical_not(mat_arr[2])
    lms = mat_arr.transpose([2, 0, 1])
    kpts = mat_arr.transpose([2, 1, 0]).tolist()
    for kpt in kpts:
        for pt in kpt:
            pt[1],pt[0] = pt[0],pt[1]



    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]

        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

        # centers.append([0, 0])
        # scales.append(0)

    return kpts, centers, scales

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def getBoundingBox(img, kpt, height, width, stride):
    x = []
    y = []

    for index in range(0,len(kpt)):
        if float(kpt[index][1]) >= 0 or float(kpt[index][0]) >= 0:
            x.append(float(kpt[index][1]))
            y.append(float(kpt[index][0]))

    x_min = int(max(min(x), 0))
    x_max = int(min(max(x), width))
    y_min = int(max(min(y), 0))
    y_max = int(min(max(y), height))

    # box = np.zeros(4)
    # box[0] = (x_min + x_max)/2
    # box[1] = (y_min + y_max)/2
    # box[2] =  x_max - x_min
    # box[3] =  y_max - y_min

    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2
    w        =  x_max - x_min
    h        =  y_max - y_min

    coord = []
    coord.append([min(int(center_y/stride),height/stride-1), min(int(center_x/stride),width/stride-1)])
    coord.append([min(int(y_min/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
    coord.append([min(int(y_min/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])
    coord.append([min(int(y_max/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
    coord.append([min(int(y_max/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])

    box = np.zeros((int(height/stride), int(width/stride), 5), dtype=np.float32)
    for i in range(5):
        # resize from 368 to 46
        x = int(coord[i][0]) * 1.0
        y = int(coord[i][1]) * 1.0
        heat_map = guassian_kernel(size_h=int(height/stride), size_w=int(width/stride), center_x=x, center_y=y, sigma=3)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        box[:, :, i] = heat_map

    return box

class LSP_Dataset_Generator:
    """
         0 = Right Ankle
         1 = Right Knee
         2 = Right Hip
         3 = Left  Hip
         4 = Left  Knee
         5 = Left  Ankle
         6 = Right Wrist
         7 = Right Elbow
         8 = Right Shoulder
         9 = Left  Shoulder
        10 = Left  Elbow
        11 = Left  Wrist
        12 = Neck
        13 = Head  Top
    """

    def __init__(self, mode, root_dir, sigma, stride, transformer=None):

        self.img_list    = read_data_file(root_dir)
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        self.stride      = stride
        self.transformer = transformer
        self.sigma       = sigma
        self.bodyParts   = [[13, 12], [12, 9], [12, 8], [8, 7], [9, 10], [7, 6], [10, 11], [12, 3], [2, 3], [2, 1], [1, 0], [3, 4], [4, 5]]

    def __getitem__(self, index):

        img_path = self.img_list[index]
        # img = np.array(cv2.imread(img_path), dtype=np.float32)
        img = cv2.imread(img_path)
        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        if self.transformer:
            img, kpt, center = self.transformer(img, kpt, center, scale)
        height, width, _ = img.shape


        heatmap = np.zeros((round(height/self.stride), round(width/self.stride), int(len(kpt)+1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height/self.stride+0.5),size_w=int(width/self.stride+0.5), center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background


        img = img.transpose([2, 0, 1])
        #img = transforms.normalize(img, 128.0,256.0)
        img = img / 255
        #img = transforms.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #img = (img - 0.5) / 0.5
        # rescale_nml  = 1 / 0.3081
        # shift_nml = -1 * 0.1307 / 0.3081
        # img = img * rescale_nml + shift_nml

        heatmap = heatmap.transpose([2, 0, 1])


        return img, heatmap

    def __len__(self):
        return len(self.img_list)


def CreateDatasetLSP(cfg):
    if cfg.IS_MODELART:
        train_dir = cfg.CACHE_INPUT + cfg.CACHE_TRAIN
        val_dir = cfg.CACHE_INPUT + cfg.CACHE_VAL
    else:
        train_dir = cfg.train_dir
        val_dir = cfg.val_dir
    if cfg.IS_DISTRIBUTE:
        train_loader = ds.GeneratorDataset(LSP_Dataset_Generator('joints_train',
                                                      train_dir,  cfg.sigma, cfg.stride,
                                                      transforms.Compose([
                                                          transforms.RandomColor(h_gain=0.8, s_gain=0.8, v_gain=0.8),
                                                          transforms.GaussianBlur(kernel_size=7,prob=0.3,sigma=5),
                                                          transforms.TypeCast(),
                                                          transforms.RandomRotate(max_degree=10),
                                                          transforms.TestResized(size=(cfg.height,cfg.width)), # H,W
                                                            transforms.RandomHorizontalFlip()
                                                                        ])
                                                             ),
                                       column_names=["img", "heatmap"],
                                       num_shards=get_group_size(),
                                       shard_id=get_rank(),
                                       shuffle=True,num_parallel_workers=cfg.workers)
    else:
        train_loader = ds.GeneratorDataset(LSP_Dataset_Generator('joints_train',
                                                                 train_dir, cfg.sigma, cfg.stride,
                                                      transforms.Compose([
                                                          transforms.RandomColor(h_gain=0.8, s_gain=0.8, v_gain=0.8),
                                                          transforms.GaussianBlur(kernel_size=7,prob=0.3,sigma=5),
                                                          transforms.TypeCast(),
                                                          transforms.RandomRotate(max_degree=10),
                                                          transforms.TestResized(size=(cfg.height,cfg.width)), # H,W
                                                            transforms.RandomHorizontalFlip()
                                                                        ])
                                                                 ),
                                           column_names=["img", "heatmap"],
                                           shuffle=True, num_parallel_workers=cfg.workers)
    val_loader = ds.GeneratorDataset(LSP_Dataset_Generator('joints_test',
                                                      val_dir, cfg.sigma, cfg.stride,
                                                      transforms.Compose([
                                                          transforms.TypeCast(),
                                                          transforms.TestResized(size=(cfg.height,cfg.width)), ]) # shape=[H,W]
                                                           ),
                                       column_names=["img", "heatmap"],
                                     shuffle=True,num_parallel_workers=cfg.workers)
    '''
    transforms_list = [
        mindspore.dataset.vision.c_transforms.Normalize(mean=[128.0, 128.0, 128.0], std=[255.0, 255.0, 255.0]),
        mindspore.dataset.vision.c_transforms.HWC2CHW()
    ] # [0,1,2] -> [2,0,1]
    train_dataset = train_loader.map(operations=transforms_list, input_columns="img")
    val_dataset = val_loader.map(operations=transforms_list, input_columns="img")
    train_dataset = train_dataset.batch(batch_size=cfg.batch_size)
    val_dataset = val_dataset.batch(batch_size=cfg.TEST_batch_size)
    '''

    train_dataset = train_loader.batch(batch_size=cfg.batch_size)
    val_dataset = val_loader.batch(batch_size=cfg.TEST_batch_size)


    return train_dataset, val_dataset

if __name__ =="__main__":
    img_path = '../data/LSP/TRAIN/im0001.jpg'
    img = np.array(cv2.resize(cv2.imread(img_path), (368, 368)), dtype=np.float32)

    img = transforms.normalize(transforms.to_tensor(img), [128.0, 128.0, 128.0],
                               [256.0, 256.0, 256.0])
    train_dataset, val_dataset = CreateDatasetLSP(cfg)
    print(train_dataset.dataset_size)
    print("dataset OK !")