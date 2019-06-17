import torch
import numpy as np
import h5py
from PIL import Image


class etl(torch.utils.data.Dataset):

    def __init__(self, split, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('age_gender.h5', 'r', driver='core')

        if self.split == 'training':
            self.train_datas = self.data['training_pixel']
            self.train_labels_age = self.data['training_label_age']
            self.train_labels_gender = self.data['training_label_gender']
            self.train_datas = np.asarray(self.train_datas)
        else:
            self.test_datas = self.data['testing_pixel']
            self.test_labels_age = self.data['testing_label_age']
            self.test_labels_gender = self.data['testing_label_gender']
            self.test_datas = np.asarray(self.test_datas)

    def __getitem__(self, index):

        if self.split == 'training':
            img, target1, target2 = self.train_datas[index], self.train_labels_age[index], self.train_labels_gender[index]
        else:
            img, target1, target2 = self.test_datas[index], self.test_labels_age[index], self.test_labels_gender[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target1, target2

    def __len__(self):
        if self.split == 'training':
            return len(self.train_datas)
        else:
            return len(self.test_datas)