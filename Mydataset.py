import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import random
from PIL import Image
import glob

class ISBICellDataloader(data.Dataset):
    def __init__(self, root=None, dataset_type='train',  cross=0, K=3, transform=None):
        self.root = root
        self.dataset_type = dataset_type
        self.transform = transform

        self.data = sorted(os.listdir(self.root + "/Image"))
        self.label = sorted(os.listdir(self.root + "/Label"))
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        
        idx = np.arange(0, len(self.data))
        idx_train = idx[np.fmod(idx, K) != cross].astype(np.int32) 
        idx_test = idx[np.fmod(idx, K) == cross].astype(np.int32) 

        if self.dataset_type=='train':
            self.datas = self.data[idx_train]
            self.labels = self.label[idx_train]
        
        else:
            self.datas = self.data[idx_test]
            self.labels = self.label[idx_test]
        
        self.data_path = []
        self.label_path = []
        for i in range(len(self.datas)):
            self.data_num = sorted(os.listdir(self.root + "/Image/{}".format(self.datas[i])))
            self.label_num = sorted(os.listdir(self.root + "/Label/{}".format(self.labels[i])))
            for j in range(len(self.data_num)):
                self.data_path.append("{}/".format(self.datas[i]) + self.data_num[j])
                self.label_path.append("{}/".format(self.labels[i]) + self.label_num[j])
            
        

    def __getitem__(self, index):
        # data
        image_name = self.root + "/Image/" + self.data_path[index]
        label_name = self.root + "/Label/" + self.label_path[index]
        image = Image.open(image_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        label = np.array(label)
        label = np.where(label[:,:,0]>=150, 0, 1)
        label = Image.fromarray(np.uint8(label))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.data_path)
        
        
class Drosophila_Dataloader(data.Dataset):
    def __init__(self, rootdir="Dataset", val_area=1, split='train', iteration_number=None, transform=None):
        self.split = split
        self.training = True if split == 'train' else False
        self.transform = transform
        filelist_train = []
        filelist_val = []
        filelist_test = []
        test_area = val_area + 1 if val_area != 5 else 1

        for i in range(1, 6):
            dataset = sorted(glob.glob(os.path.join(rootdir, "data", "5-fold", f"Area_{i}", "*.npy")))
            if i == val_area:
                filelist_test = filelist_test + dataset
            elif i == test_area:
                filelist_val = filelist_val + dataset
            else:
                filelist_train = filelist_train + dataset
        

        if split == 'train':
            self.filelist = filelist_train
        elif split == 'val':
            self.filelist = filelist_val
        elif split == 'test':
            self.filelist = filelist_test

        if self.training:
            self.number_of_run = 1
            self.iterations = iteration_number
        else:
            self.number_of_run = 16
            self.iterations = None

        print(f'val_area : {val_area} test_area : {test_area} ', end='')
        print(f"{split} files : {len(self.filelist)}")

    def __getitem__(self, index):
        # print(index)
        if self.training:
            index = random.randint(0, len(self.filelist) - 1)
            dataset = self.filelist[index]

        else:
            dataset = self.filelist[index // self.number_of_run]

        # load files
        filename_data = os.path.join(dataset)
        inputs = np.load(filename_data)
        #4(12)枚の大きな画像から256*256の画像を生成
        #4(12)*(1024/256)*(1024/256)=64(192)
        #実質64(192)枚学習
        # split features labels
        if self.training:
            x = random.randint(0, inputs.shape[0] - 256)
            y = random.randint(0, inputs.shape[0] - 256)
        else:
            x = index % self.number_of_run//4 * 256
            y = index % self.number_of_run % 4 * 256

        features = inputs[:,:,0:1].astype(np.float32)#transpose(2, 0, 1)
        features = np.repeat(features, 3, 2)
        #features /= 255.0
        
        labels = inputs[:,:, -1].astype(int)
        features = Image.fromarray(np.uint8(features))
        labels = Image.fromarray(np.uint8(labels))
        if self.transform:
            fts, lbs = self.transform(features, labels)
        #fts = torch.from_numpy(features).float()
        #lbs = torch.from_numpy(labels).long()

        return fts, lbs

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

    def get_class_count(self):
        label = np.array([np.load(data_path)[:, :, -1] for data_path in self.filelist])
        
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]

        return label_count

