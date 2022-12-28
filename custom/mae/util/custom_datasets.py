# TODO
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pathlib import Path
import pandas as pd

class MultiFocalPlaneDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.npy_files = list(Path(folder).glob("**/*.npy"))
        self.focal_plane_number = 5
        self.transform = transform
        self.fit_into_ram = False
        self.num_frame = 209
        self.reset()

        # self.all_frames = [np.load(file) for file in self.npy_files]
    def __len__(self):
        return len(self.npy_files)*self.num_frame

    def reset(self):
        self.rest_files = self.npy_files
        self.cached_files = []
        self.cached_frames = []
        self.get_random_100_npy_files_from_rest()

    def get_random_100_npy_files_from_rest(self):
        self.cached_files = np.random.choice(self.rest_files, min(100, len(self.rest_files)), replace=False)
        self.rest_files = list(set(self.rest_files) - set(self.cached_files))
        self.cached_frames = [np.load(file) for file in self.cached_files]
        self.used_idx_dict = {} # {used_file_idx: [used_frame_idx]}
        self.used_total_frame_num = 0
        print(f're-get random 100 npy files from the rest files({len(self.rest_files)})!')

    def _is_all_frames_of_the_file_used(self, file_idx):
        if file_idx not in self.used_idx_dict:
            self.used_idx_dict[file_idx] = []

        all_frames_of_the_file_used = len(self.used_idx_dict[file_idx]) == self.num_frame
        return all_frames_of_the_file_used

    def _target_frame_used(self, file_idx, frame_idx):
        is_target_frame_used = frame_idx in self.used_idx_dict[file_idx]
        return is_target_frame_used

    def _mark_target_frame_used(self, file_idx, frame_idx):
        self.used_idx_dict[file_idx].append(frame_idx)
        self.used_total_frame_num += 1

    def __getitem__(self, idx):
        if len(self.rest_files) == 0:
            self.reset()

        if self.used_total_frame_num == 100*self.num_frame:
            self.get_random_100_npy_files_from_rest()

        # frames = self.all_frames[idx]
        file_idx = idx%100
        while self._is_all_frames_of_the_file_used(file_idx):
            file_idx = (file_idx+1)%100 # to optimize

        frame_idx = idx%self.num_frame
        while self._target_frame_used(file_idx, frame_idx):
            frame_idx = (frame_idx+1)%209 # to optimizer
        
        self._mark_target_frame_used(file_idx, frame_idx)
        frame = None
        frame = self.transform(image=self.cached_frames[file_idx][frame_idx])['image']
        return frame, []


def broadcast_to_3_ch(img):
    if len(np.squeeze(img).shape) == 2:
        h, w = np.squeeze(img).shape
        img = np.broadcast_to(np.expand_dims(np.squeeze(img), -1), (h, w, 3))
    return img

# class SingleFocalPlaneDataset(Dataset):
#     def __init__(self, folder, transform=None):
#         self.folder = folder
#         self.npy_files = list(Path(folder).glob("**/*.npy"))
#         self.focal_plane_number = 5
#         self.transform = transform
#         self.fit_into_ram = False
#         self.num_frame = 209
#         self.reset()

#         # self.all_frames = [np.load(file) for file in self.npy_files]
#     def __len__(self):
#         return len(self.npy_files)*self.num_frame

#     def reset(self):
#         self.rest_files = self.npy_files
#         self.cached_files = []
#         self.cached_frames = []
#         self.get_random_100_npy_files_from_rest()

#     def get_random_100_npy_files_from_rest(self):
#         self.cached_files = np.random.choice(self.rest_files, min(100, len(self.rest_files)), replace=False)
#         self.rest_files = list(set(self.rest_files) - set(self.cached_files))
#         self.cached_frames = [broadcast_to_3_ch(np.load(file)) for file in self.cached_files]
#         self.used_idx_dict = {} # {used_file_idx: [used_frame_idx]}
#         self.used_total_frame_num = 0
#         print(f're-get random 100 npy files from the rest files({len(self.rest_files)})!')

#     def _is_all_frames_of_the_file_used(self, file_idx):
#         if file_idx not in self.used_idx_dict:
#             self.used_idx_dict[file_idx] = []

#         all_frames_of_the_file_used = len(self.used_idx_dict[file_idx]) == self.num_frame
#         return all_frames_of_the_file_used

#     def _target_frame_used(self, file_idx, frame_idx):
#         is_target_frame_used = frame_idx in self.used_idx_dict[file_idx]
#         return is_target_frame_used

#     def _mark_target_frame_used(self, file_idx, frame_idx):
#         self.used_idx_dict[file_idx].append(frame_idx)
#         self.used_total_frame_num += 1

#     def __getitem__(self, idx):
#         if len(self.rest_files) == 0:
#             self.reset()

#         if self.used_total_frame_num == 100*self.num_frame:
#             self.get_random_100_npy_files_from_rest()

#         # frames = self.all_frames[idx]
#         file_idx = idx%100
#         while self._is_all_frames_of_the_file_used(file_idx):
#             file_idx = (file_idx+1)%100 # to optimize

#         frame_idx = idx%self.num_frame
#         while self._target_frame_used(file_idx, frame_idx):
#             frame_idx = (frame_idx+1)%209 # to optimizer
        
#         self._mark_target_frame_used(file_idx, frame_idx)
#         frame = None
#         frame = self.transform(image=self.cached_frames[file_idx][frame_idx])['image']
#         return frame, []

def _normalize_emr_data(df, seed):
    mean_std_cols = ['age']
    min_max_cols = []
    
    for col in mean_std_cols:
        col = f'hetero_{col}'
        mean = df.loc[df[f'seed{seed}'] == 'train', col].mean()
        std = df.loc[df[f'seed{seed}'] == 'train', col].std()
        df[col] = (df[col] - mean)/std
        
    return df


class CustomHeterogeneousDataset(Dataset):
    def __init__(self, is_train, args=None, transform=None, resize_when_read=True):
        self.df = pd.read_excel(args.anno_path)
        self.df = _normalize_emr_data(self.df, args.seed)
        if is_train:
            self.df = self.df[self.df[f'seed{args.seed}'] == 'train']
        else:
            self.df = self.df[self.df[f'seed{args.seed}'] == 'test']
        self.resize_when_read = resize_when_read
        self.input_size = args.input_size
        self.images = []
        self.image_paths = self.df['path'].to_list()
        self._read_images()
        self.labels = self.df[args.label].to_list()
        self.statistical_data = np.array(self.df[[col for col in self.df if 'hetero_' in col]])
        self.transform = transform
        print(self.df.shape, len(self.image_paths), len(self.labels), transform)
    
    def _read_images(self):
        for i, path in enumerate(self.image_paths):
            image = Image.open(path).convert("RGB")
            if self.resize_when_read:
                image = image.resize((self.input_size, self.input_size))
            self.images.append(image)
            if i % 1000 == 0:
                print(i, len(self.image_paths))
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if len(self.images) > 0:
            image = self.images[index]
        else:
            image = Image.open(self.image_paths[index])
            image = image.convert("RGB")
            
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        statistical_data = torch.tensor(self.statistical_data[index]).float()
        return (image, statistical_data), label

class CustomImageDataset(Dataset):
    def __init__(self, is_train, args=None, transform=None):
        self.df = pd.read_excel(args.anno_path)
        if is_train:
            self.df = self.df[self.df[f'seed{args.seed}'] == 'train']
        else:
            self.df = self.df[self.df[f'seed{args.seed}'] == 'test']
        
        self.images = []
        self.image_paths = self.df['path'].to_list()
        # self._read_images()
        self.labels = self.df[args.label].to_list()
        self.transform = transform
        print(self.df.shape, len(self.image_paths), len(self.labels), transform)
    
    def _read_images(self):
        for i, path in enumerate(self.image_paths):
            self.images.append(Image.open(path).convert("RGB"))
            if i % 10000 == 0:
                print(i, len(self.image_paths))
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if len(self.images) > 0:
            image = self.images[index]
        else:
            image = Image.open(self.image_paths[index])
            image = image.convert("RGB")
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label