import torch
import shutil
import os
import splitfolders
import pytorch_lightning as pl

from PIL import Image
from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms


class DiffSet(Dataset):
    def __init__(self, absolute_path: str,
                       path2data: str, 
                       name_dataset: str = "MNIST",
                       picture_size: [int, int] = [32, 32],
                       transform: transforms.Compose = None):

        self.absolute_path = absolute_path
        self.path2data = path2data
        self.name_dataset = name_dataset
        self.picture_size = picture_size

        if transform is None:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((self.picture_size[0], self.picture_size[1])),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        if name_dataset in datasets:
            transform = transforms.Compose([transforms.ToTensor()])

            train_dataset = datasets[name_dataset](
                self.absolute_path + "/Data", download=True, train=(self.path2data.split("/")[-2] == "train"), transform=transform
            )

            self.dataset_len = len(train_dataset.data)

            if name_dataset == "MNIST" or name_dataset == "Fashion":
                pad = transforms.Pad(2)
                data = pad(train_dataset.data)
                data = data.unsqueeze(3)
                self.depth = 1
                # self.size = 32
            elif name_dataset == "CIFAR":
                data = torch.Tensor(train_dataset.data)
                self.depth = 3
                # self.size = 32
            self.input_seq = ((data / 255.0) * 2.0) - 1.0
            self.input_seq = self.input_seq.moveaxis(3, 1)

            def get_item(item):
              return self.input_seq[item]
            self.get_item = get_item
        else:
            self.list_files = os.listdir(self.path2data)
            self.dataset_len = len(self.list_files)

            current_image_name = self.list_files[0]
            path_to_image = os.path.join(self.path2data, current_image_name)
            current_image = Image.open(path_to_image)
            self.depth = len(current_image.getbands())


            def get_item(index):
                current_image_name = self.list_files[index]
                path_to_image = os.path.join(self.path2data, current_image_name)
                current_image = Image.open(path_to_image)
                image = self.transform(current_image)
                return image
            self.get_item = get_item

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.get_item(item)

# Datamodule class
class DiffDataModule(pl.LightningDataModule):
    def __init__(self,
                 absolute_path: str,
                 name_dataset: str,
                 path2data: str,
                 batch_size: int = 32,
                 picture_size: [int, int] = [256, 256],
                 shuffle: bool = True,
                 transform: transforms.Compose = None):
        super().__init__()
        self.absolute_path = absolute_path
        self.name_dataset = name_dataset
        self.path2data = path2data
        self.sorted_folder_data_path = self.absolute_path + "/Data/SortedData"
        self.batch_size = batch_size
        self.picture_size = picture_size
        self.shuffle = shuffle
        self.transform = transform

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }
        # if not (name_dataset in datasets):
        #   self.create_samples()

        self.train = DiffSet(self.absolute_path, 
                             self.sorted_folder_data_path + "/train/" + self.path2data.split("/")[-1],
                             self.name_dataset,
                             self.picture_size, 
                             self.transform)

    def create_samples(self):
        temp_folder_path = self.absolute_path + "/Data/RawData/TempFold"
        if not os.path.exists(temp_folder_path):
            os.makedirs(temp_folder_path)

        shutil.move(self.absolute_path + self.path2data, temp_folder_path)

        if not os.path.exists(self.sorted_folder_data_path):
            os.makedirs(self.sorted_folder_data_path)
        self.clean_folder(self.sorted_folder_data_path)

        splitfolders.ratio(temp_folder_path, output=self.sorted_folder_data_path, seed=7, ratio=(.7, 0.15, 0.15))

        shutil.move(temp_folder_path + "/" + self.path2data.split("/")[-1], self.absolute_path + self.path2data)
        shutil.rmtree(temp_folder_path)

    def clean_folder(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            # self.train = DiffSet(self.absolute_path, self.sorted_folder_data_path + "/train", self.name_dataset,
            #                                         self.picture_size, self.transform)

            self.validation = DiffSet(self.absolute_path, 
                                      self.sorted_folder_data_path + "/val/" + self.path2data.split("/")[-1],
                                      self.name_dataset,
                                      self.picture_size, 
                                      self.transform)

        if stage in (None, "test"):
            self.test = DiffSet(self.absolute_path, 
                                self.sorted_folder_data_path + "/test/" + self.path2data.split("/")[-1],
                                self.name_dataset, 
                                self.picture_size, 
                                self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, num_workers=12)