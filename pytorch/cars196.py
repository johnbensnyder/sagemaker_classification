import os
import shutil
from pathlib import Path
import scipy.io
import torch
import torch.utils.data as data
import torchvision
from tqdm.auto import tqdm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity

class Cars196(ImageFolder):
    
    base_folder_devkit = 'devkit'
    url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    filename_devkit = 'cars_devkit.tgz'
    tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'
    
    base_folder_trainims = 'cars_train'
    url_trainims = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    filename_trainims = 'cars_ims_train.tgz'
    tgz_md5_trainims = '065e5b463ae28d29e77c1b4b166cfe61'
    
    base_folder_testims = 'cars_test'
    url_testims = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    filename_testims = 'cars_ims_test.tgz'
    tgz_md5_testims = '4ce7ebf6a94d07f1952d94dd34c4d501'
    
    url_testanno = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
    filename_testanno = 'cars_test_annos_withlabels.mat'
    mat_md5_testanno = 'b0a2b23655a3edd16d84508592a98d10'
    
    filename_trainanno = 'cars_train_annos.mat'
    
    test_list = []
    num_classes = 196
    
    def __init__(self, data_dir, train=False, transform=None, target_transform=None, download=False, **kwargs):
        
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, self.base_folder_trainims if train else self.base_folder_testims)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.train = train
        
        if download:
            # download devkit
            self.url, self.filename, self.tgz_md5 = self.url_devkit, self.filename_devkit, self.tgz_md5_devkit
            self.download()
            
            if train:
                # download train data
                self.url, self.filename, self.tgz_md5 = self.url_trainims, self.filename_trainims, self.tgz_md5_trainims
                self.download(organize_files=True)
            else:
                # download test data
                self.url, self.filename, self.tgz_md5 = self.url_testims, self.filename_testims, self.tgz_md5_testims
                self.download(organize_files=True)

        self.annos = scipy.io.loadmat(os.path.join(self.data_dir, self.base_folder_devkit,
                                                   "cars_{0}_annos.mat".format("train" if self.train else "test")))
        
        self.metas = scipy.io.loadmat(os.path.join(self.data_dir, self.base_folder_devkit,
                                                   "cars_meta.mat"))
        
        super().__init__(self.root, transform=self.transform, target_transform=self.target_transform, **kwargs)
            
    def download(self, organize_files=False):
        if check_integrity(os.path.join(self.data_dir, self.filename), self.tgz_md5):
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.data_dir, filename=self.filename, md5=self.tgz_md5)
        if organize_files:
            self.organize_dir()
    
    def organize_dir(self):
        image_root_path = Path(self.root)
        print('Organizing files to imagefile format')
        for anno in tqdm(self.annos['annotations'][0]):
            new_dir_path = image_root_path.joinpath(str(anno[4][0,0]).zfill(3))
            new_dir_path.mkdir(exist_ok=True)
            shutil.move(image_root_path.joinpath(anno[5][0]),
                        new_dir_path.joinpath(anno[5][0]))
