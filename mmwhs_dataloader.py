import torch
import glob
import os
import random
import itertools

from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, MapLabelValue
from torchvision.transforms import Normalize


class MMWHS(Dataset):
  def __init__(self, args, labels, fold):
    self.data_dirs = args.data_dir_s # source data
    self.data_dirt = args.data_dir_t # target data
    self.target_labels = labels

    self.all_source_images = sorted(glob.glob(os.path.join(self.data_dirs, "images/case_10*")))
    self.all_source_labels = sorted(glob.glob(os.path.join(self.data_dirs, "labels/case_10*")))
    self.all_target_images = sorted(glob.glob(os.path.join(self.data_dirt, "images/case_10*")))
    self.all_target_labels = sorted(glob.glob(os.path.join(self.data_dirt, "labels/case_10*")))

    images_source = [glob.glob(self.all_source_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images_source = sorted(list(itertools.chain.from_iterable(images_source)))
    labels_source = [glob.glob(self.all_source_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels_source = sorted(list(itertools.chain.from_iterable(labels_source)))
    images_target = [glob.glob(self.all_target_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images_target = sorted(list(itertools.chain.from_iterable(images_target)))
    labels_target = [glob.glob(self.all_target_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels_target = sorted(list(itertools.chain.from_iterable(labels_target)))
        
    self.source_size = len(self.images_source)
    self.target_size = len(self.images_target)
    self.dataset_size = max(self.source_size, self.target_size)

    if self.target_size is not self.source_size:
      print("something is wrong here")
      print(self.target_size)
      print(self.source_size)


    self.transforms_seg = Compose(
            [
              LoadImage(),
              MapLabelValue(orig_labels=[1, 2, 3, 4, 5 , 6 , 7], target_labels=self.target_labels),
              MapLabelValue(orig_labels=[421], target_labels=[0]),
            ])
  
    self.transform_img = Compose(
            [
              LoadImage(),
              #Normalize(mean=[0.5], std=[0.5])
            ])


  def __getitem__(self, index):
    
    image_s = self.transform_img(self.images_source[index])
    label_s = self.transforms_seg(self.labels_source[index])
    
    image_t = self.transform_img(self.images_target[index])
    label_t = self.transforms_seg(self.labels_target[index])
  
    return image_s, label_s, image_t, label_t

  def __len__(self):
    return self.dataset_size
  

class MMWHS_single(Dataset):
  def __init__(self, args, fold, labels = [1, 0, 0, 0, 0, 0, 0]):
    self.data_dir = args.data_dir 
    self.target_labels = labels

    self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
    self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))

    images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
    self.test_images = sorted(list(itertools.chain.from_iterable(images)))
    labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.test_labels = sorted(list(itertools.chain.from_iterable(labels)))
    
    self.dataset_size = len(self.test_images)

    self.transforms_seg = Compose(
            [
              LoadImage(),
              MapLabelValue(orig_labels=[1, 2, 3, 4, 5, 6, 7], target_labels=self.target_labels),
              MapLabelValue(orig_labels=[421], target_labels=[0])
            ])
    
    self.transform_img = Compose(
            [
              LoadImage(),
            ])


  def __getitem__(self, index):
    image = self.transform_img(self.test_images[index])
    label = self.transforms_seg(self.test_labels[index])

    return image, label

  def __len__(self):
    return self.dataset_size