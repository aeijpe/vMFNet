import glob
import os
import itertools

from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged, MapLabelValued

class CHAOS_single(Dataset):
  def __init__(self, data_dir, fold, labels = [1, 0, 0, 0]):
    self.data_dir = data_dir
    self.target_labels = labels

    self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_*")))
    self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_*")))

    images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images = sorted(list(itertools.chain.from_iterable(images)))
    labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels = sorted(list(itertools.chain.from_iterable(labels)))

    
    self.dataset_size = len(self.images)

    self.transform_dict = Compose(
        [
            LoadImaged(keys=["img", "seg"],),
            MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4], target_labels=self.target_labels)
        ]
    )
        
    
    self.data = [{"img": img, "seg": seg} for img, seg in zip(self.images, self.labels)] 


  def __getitem__(self, index):
    data_point = self.transform_dict(self.data[index])
    image = data_point["img"]
    label = data_point["seg"] 

    return image.unsqueeze(0), label.unsqueeze(0)

  def __len__(self):
    return self.dataset_size
  
class MMWHS_single(Dataset):
  def __init__(self, data_dir, fold, labels = [1, 0, 0, 0, 0]):
    self.data_dir = data_dir
    self.target_labels = labels

    self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
    self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))

    images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images = sorted(list(itertools.chain.from_iterable(images)))
    labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels = sorted(list(itertools.chain.from_iterable(labels)))

    
    self.dataset_size = len(self.images)

    self.transform_dict = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5], target_labels=self.target_labels),
            MapLabelValued(keys=["seg"], orig_labels=[421], target_labels=[0]),
        ]
    )
        
    self.data = [{"img": img, "seg": seg} for img, seg in zip(self.images, self.labels)] 


  def __getitem__(self, index):
    data_point = self.transform_dict(self.data[index])
    image = data_point["img"]
    label = data_point["seg"] 

    return image, label

  def __len__(self):
    return self.dataset_size