#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:44:38 2020

@author: bdrhn9
"""
import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset  # For custom datasets
from PIL import Image
import torch
import cv2
import pandas
import shutil
import random
import torchvision
from sklearn.model_selection import train_test_split
from skimage import transform as sk_transform

class ESOGU_Faces(Dataset):
    def __init__(self, folder_path='./data/esogu_faces',meta_path='./data/esogu_faces_gt.pkl',split='train',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.meta_list = pickle.load(open(meta_path,'rb'))[split]
        self.folder_path = folder_path
        # Calculate len
        self.data_len = len(self.meta_list)
        self.transform = transform
        self.num_classes = len(np.unique(np.asarray(self.meta_list)[:,0]))
    def __getitem__(self, index):
        path_gt = self.meta_list[index]
        # Open image
        img = Image.open(os.path.join(self.folder_path,'%d/%d/%d.jpg'%(path_gt[0],path_gt[1],path_gt[2]))).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(path_gt)
        return (img, label)

    def __len__(self):
        return self.data_len

class COX_Faces(Dataset):
    def __init__(self, folder_path='./data/GrayLevels',meta_path='./data/Cox23_1.pkl',split='train',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.meta_list = pickle.load(open(meta_path,'rb'))[split]
        self.folder_path = folder_path
        # Calculate len
        self.split=split
        self.data_len = len(self.meta_list)
        self.transform = transform
    def __getitem__(self, index):
        path_gt = self.meta_list[index]
        if (path_gt[1] == 1 or path_gt[1] == 0) and len(path_gt) == 4:
            img = Image.open(os.path.join(self.folder_path,'%d/%d/%d.jpg'%(path_gt[0],path_gt[1]+1,path_gt[2]))).convert('RGB')
        else:
            img = Image.open(os.path.join(self.folder_path,'%d/%d/%d.jpg'%(path_gt[0],path_gt[1],path_gt[2]))).convert('RGB')

        # Open image
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(path_gt)
        return (img, label)

    def __len__(self):
        return self.data_len
    
class Vgg_Face2_original(Dataset):
    def __init__(self, folder_path='./data/Vgg-Aligned',meta_path='./data/Vgg-Aligned/train/train.csv',split='train',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.meta_list = pd.read_csv(meta_path)
        self.folder_path = folder_path
        self.num_classes = np.unique((self.meta_list['Folder_Name'])).size
        # Calculate len
        self.split=split
        self.data_len = len(self.meta_list)
        self.transform = transform
    def __getitem__(self, index):
        path_gt = [int(self.meta_list['Folder_Name'][index][1:]),0, int(self.meta_list['Img_Name'][index][:-4]),0]
        img = Image.open(os.path.join(self.folder_path,'%s/%s/%s/%s'%(self.split,self.split,self.meta_list['Folder_Name'][index], self.meta_list['Img_Name'][index]))).convert('RGB')
        
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(path_gt)
        # label = path_gt
        return (img, label)

    def __len__(self):
        return self.data_len

class Vgg_Face2(Dataset):
    def __init__(self, folder_path='./data/Vgg-Aligned',meta_path='./data/Vgg-Aligned/train/train.csv', label_path='./data/3labels_VggFacesTrain.npy',split='train',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.meta_list = pd.read_csv(meta_path)
        self.meta_label = np.load(label_path)
        self.folder_path = folder_path
        self.num_classes = len(np.unique(self.meta_label[:,0]))
        # Calculate len
        self.split=split
        self.data_len = len(self.meta_list)
        self.transform = transform
    def __getitem__(self, index):
        path_gt = self.meta_label[index]
        img = Image.open(os.path.join(self.folder_path,'%s/%s/%s/%s'%(self.split,self.split,self.meta_list['Folder_Name'][index], self.meta_list['Img_Name'][index]))).convert('RGB')
        
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(path_gt)
        return (img, label)

    def __len__(self):
        return self.data_len

class VGGFace2_AlignedArc(Dataset):
    def __init__(self, folder_path='./data/Vgg-Aligned',meta_path='./data/3labels_VggFacesArc_centerloss.npy',split='train',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.meta_label = np.load(meta_path)
        self.path_list = np.load('./data/3path_VggFacesArc_centerloss.npy').tolist()
        self.num_classes = 8631
        # Calculate len
        self.split=split
        self.data_len = len(self.meta_label)
        self.transform = transform
    def __getitem__(self, index):
        
        img = Image.open(self.path_list[index]).convert('RGB')
        
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(self.meta_label[index],dtype=int)
        # label = path_gt
        return (img, label)

    def __len__(self):
        return self.data_len    

class PaSC(Dataset):
    def __init__(self, folder_path ='./data/pasc/pasc-align-final-train',meta_path='./data/pasc/Pasc-Train.csv',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.df = pandas.read_csv(meta_path)
        classes = []
        for folder_name in self.df['Folder_Name']:
            classes.append(folder_name.split('d')[0])
        self.classes = set(classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.num_classes = len(self.classes)
        self.data_len = len(self.df)
        self.transform = transform
        self.folder_path = folder_path
        
    def __getitem__(self, index):
        file_path = self.df['MAT_PATH'][index]
        img = Image.open(os.path.join(self.folder_path,file_path))
        label = self.class_to_idx[file_path.split('d')[0]]
        label = np.array(label)
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        # label = path_gt
        return (img, label, file_path)

    def __len__(self):
        return self.data_len

class PaSC_Folder(Dataset):
    def __init__(self, folder_path ='./data/pasc/pasc-align-final',meta_path='./data/pasc/Pasc-Train.csv',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.df = pandas.read_csv(meta_path)
        classes = []
        for folder_name in self.df['Folder_Name']:
            classes.append(folder_name.split('d')[0])
        self.classes = set(classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.num_classes = len(self.classes)
        self.data_len = len(self.df)
        self.transform = transform
        self.folder_path = folder_path
        
    def __getitem__(self, index):
        file_path = self.df['MAT_PATH'][index]
        img = Image.open(os.path.join(self.folder_path,file_path))
        label = self.class_to_idx[file_path.split('d')[0]]
        label = np.array(label)
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        # label = path_gt
        return (img, label, file_path)

    def __len__(self):
        return self.data_len

class Clustured_Dataset(Dataset):
    def __init__(self, folder_path ='./data/pasc/pasc-align-final-train',meta_path='data/pasc/test_meta.npy',transform=None,file_path=False,first_n_class=None,transform_aux=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.data = np.load(meta_path)
        if(first_n_class):
            print('Initialized with first %d class with just first subset'%first_n_class)
            self.data = self.data[:np.nonzero(self.data['class']==first_n_class)[0][0]]
            # self.data = self.data[self.data['subset']==0]
        self.num_classes = len(np.unique(self.data['class']))
        self.num_subset = len(np.unique(self.data['subset']))
        self.num_subcluster = len(np.unique(self.data['subcluster']))
        
        self.data_len = len(self.data)
        
        self.transform = transform
        self.transform_aux = transform_aux
        self.folder_path = folder_path
        self.file_path = file_path
    
    def __getitem__(self, index):
        file_path, cls_idx, subset_idx, subcluster_idx = self.data[index]
        img = Image.open(os.path.join(self.folder_path,file_path)).convert('RGB')
        label = np.array((cls_idx,subset_idx,subcluster_idx))

        if self.transform is not None:
            img = self.transform(img)
        
        if self.transform_aux is not None:
            img = self.transform_aux(img)
        
        if(self.file_path):
            return (img, label, file_path)
        else:
            return (img, label)

    def __len__(self):
        return self.data_len

class MNIST_3Class(Dataset):
    def __init__(self, transform=None,train=True):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        if(train):
            data_file = "training.pt"
        else:
            data_file = "test.pt"
        self.data, self.targets = torch.load(os.path.join("/home/mlcv/bdrhn9_ws/cvpr2021/ddfm-face/data/mnist/MNIST/processed", data_file))
        # indices = np.argwhere((self.targets==0) | (self.targets==1) |(self.targets==2))[0]
        
        # if(train):
        #     indices = indices[:-5400]
        
        if(train):
            desired_classes = [0,1,2]
            indlist = list()
            for class_no in desired_classes:
                indlist.append(np.argwhere(self.targets==class_no).tolist()[0])
            indices = np.concatenate(indlist)
        else:
            desired_classes = [0,1,2,3]
            indlist = list()
            for class_no in desired_classes:
                indlist.append(np.argwhere(self.targets==class_no).tolist()[0])
            indices = np.concatenate(indlist)
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        self.num_classes = len(np.unique(self.targets))
        self.num_subset = 1
        self.num_subcluster = 1
        self.data_len = len(self.data)
        
        self.transform = transform

    def __getitem__(self, index):
        cls_idx = self.targets[index]
        img = Image.fromarray(np.uint8(self.data[index])).convert('RGB')
        label = np.array((cls_idx,0,0))

        if self.transform is not None:
            img = self.transform(img)
        
        return (img, label)

    def __len__(self):
        return self.data_len

class CIFAR10_3Class(Dataset):
    def __init__(self, transform=None,train=False):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        if(train):
            data=torchvision.datasets.CIFAR10('/home/mlcv/bdrhn9_ws/cvpr2021/ddfm-face/data/cifar',download=True,train=True)
        else:
            data=torchvision.datasets.CIFAR10('/home/mlcv/bdrhn9_ws/cvpr2021/ddfm-face/data/cifar',download=True,train=False)
            
        self.data, self.targets = data.data, np.array(data.targets)
        indices = np.squeeze(np.argwhere((self.targets==0) | (self.targets==1) |(self.targets==2)))
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        self.num_classes = len(np.unique(self.targets))
        self.num_subset = 1
        self.num_subcluster = 1
        self.data_len = len(self.data)
        
        self.transform = transform

    def __getitem__(self, index):
        cls_idx = self.targets[index]
        img = Image.fromarray(np.uint8(self.data[index])).convert('RGB')
        label = np.array((cls_idx,0,0))

        if self.transform is not None:
            img = self.transform(img)
        
        return (img, label)

    def __len__(self):
        return self.data_len

class IJBA(Dataset):
    def __init__(self, folder_path ='./data/ijba/IJBA-Aligned-112',meta_path='./data/ijba/META/split1/gallery1.csv',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.df = pandas.read_csv(meta_path)
        self.data_len = len(self.df)
        self.transform = transform
        self.folder_path = folder_path
        
    def __getitem__(self, index):
        file_path = self.df['FINAL_NAME'][index]
        
        img = Image.open(os.path.join(self.folder_path,file_path))
        label = np.array((self.df['TEMPLATE_ID'][index],self.df['SUBJECT_ID'][index],self.df['MEDIA_ID'][index]))
        # Open image
        if self.transform is not None:
            img = self.transform(img)
        # label = path_gt
        return (img, label, file_path)

    def __len__(self):
        return self.data_len

# class IJBC(Dataset):
#     def __init__(self, folder_path,meta_path, landmarks_path, transform=None):
#         """
#         A dataset example where the class is embedded in the file names
#         This data example also does not use any torch transforms
#         Args:
#             folder_path (string): path to image folder
#         """
#         # Get image list
#         with open(landmarks_path) as f:
#             self.landmarks = f.readlines()
        
#         self.df = pandas.read_csv(meta_path)
#         self.data_len = len(self.df)
#         self.transform = transform
#         self.folder_path = folder_path
#         self.src = np.array([[30.2946, 51.6963],
#                         [65.5318, 51.5014],
#                         [48.0252, 71.7366],
#                         [33.5493, 92.3655],
#                         [62.7299, 92.2041] ], dtype=np.float32 )
#         self.src[:,0] += 8.0

#         self.image_size = (112,112)
#     def __getitem__(self, index):
                
#         file_path = self.df['FILENAME'][index].split("/")[-1] # to remove "img/" preword
#         name_wo_ext = file_path.split('.')[0]
#         name_landmark_score = self.landmarks[int(name_wo_ext)-1].strip().split(' ')
#         name, landmark_list, score = name_landmark_score[0],name_landmark_score[1:-1], name_landmark_score[-1]
#         assert name==file_path
#         landmark = np.array([float(x) for x in landmark_list], dtype=np.float32)
#         landmark = landmark.reshape( (5,2) )

#         rimg = cv2.imread(os.path.join(self.folder_path,file_path))
#         # pil_image=Image.open(os.path.join(self.folder_path,file_path))
#         # pil_image_np= np.asarray(pil_image)

#         # from_cv2 = Image.fromarray(rimg)
#         # from_pil_np = Image.fromarray(pil_image_np)

#         label = np.array((self.df['TEMPLATE_ID'][index],self.df['SUBJECT_ID'][index],self.df['SIGHTING_ID'][index]))
        
#         assert landmark.shape[0]==68 or landmark.shape[0]==5
#         assert landmark.shape[1]==2
#         if landmark.shape[0]==68:
#             landmark5 = np.zeros( (5,2), dtype=np.float32 )
#             landmark5[0] = (landmark[36]+landmark[39])/2
#             landmark5[1] = (landmark[42]+landmark[45])/2
#             landmark5[2] = landmark[30]
#             landmark5[3] = landmark[48]
#             landmark5[4] = landmark[54]
#         else:
#             landmark5 = landmark
#         tform = sk_transform.SimilarityTransform()
#         tform.estimate(landmark5, self.src)
#         M = tform.params[0:2,:]
#         img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img = np.transpose(img, (2,0,1)) #3*112*112, RGB
#         img_pil = Image.fromarray(img)
#         # Open image
#         if self.transform is not None:
#             img_pil = self.transform(img_pil)
#         # label = path_gt
#         return (img_pil, label, file_path)

#     def __len__(self):
#         return self.data_len


class IJBC(Dataset):
    def __init__(self, folder_path ='/home/bdrhn9/workspace/IJB_release/IJBC/aligned112',meta_path='/home/bdrhn9/workspace/IJB_release/IJBC/meta_original/meta/ijbc_g1.npy',transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        data = np.load(meta_path)

        self.names = data['names']
        self.tids = data['tids']
        self.mids = data['mids']
        self.sids =data['sids']
      

        self.data_len = len(data)
        self.transform = transform
        self.folder_path = folder_path

    def __getitem__(self, index):

        label = np.array((self.tids[index],self.sids[index],self.mids[index]))

        img = Image.open(os.path.join(self.folder_path,self.names[index]))
        
        # Open image
        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return self.data_len

# oguface = ESOGU_Faces()
# oguface.__getitem__(1)

# cox = COX_Faces()
# cox.__getitem__(1)

# all_labels = []
# vgg2 = Vgg_Face2_original(meta_path='./data/Vgg-Aligned/test/test.csv', split='test')
# vgg2.__getitem__(0)

# for i, v in enumerate(vgg2):
#     all_labels.append(v[1])

# all_labels_cat = np.asasrray(all_labels)
    
# vggface_data = VGGFace2_AlignedArc()
# vggface_data.__getitem__(0)

# pasc_folder_data = PaSC_Folder()
# pasc_folder_data.__getitem__()
# ijba = IJBA()
# ijba.__getitem__(0)

# cifar_3class=CIFAR10_3Class()

# mnist_3class=MNIST_3Class()
