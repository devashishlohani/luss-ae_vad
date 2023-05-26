import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random
from PIL import Image
from torch.nn import functional as F
from natsort import natsorted
#rng = np.random.RandomState(2020)

import torch
import torch.nn as nn
from einops import rearrange

def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((x - shift_x) + 1) / 2

def rand_2d_tensor(th=0.5, n=3, d=3):
    rand_mat = torch.rand(n, d)
    k = round(th * d)
    k_th_quant = torch.topk(rand_mat, k, largest = False)[0][:,-1:]
    bool_tensor = rand_mat <= k_th_quant
    desired_tensor = torch.where(bool_tensor,torch.tensor(1),torch.tensor(0))
    return desired_tensor


def linear(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight.cuda())
    else:
        return F.linear(input, weight.cuda(), bias.cuda())

def np_load_frame(filename, resize_height, resize_width, grayscale=False, normalize=True):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if grayscale:
        image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    if(normalize):
        image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def img_normalize(img):
    # normalize img from [0, 255] to [-1, 1].
    return (img / 127.5) - 1.0

class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=9,
                 img_extension='.jpg', dataset='ped2', jump=[2], hold=[2], return_normal_seq=False, mask_per=0, patch_size=16, tube_mask=False):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames

        self.extension = img_extension
        self.dataset = dataset
        self.mask_per = mask_per
        self.psize = patch_size

        self.setup()
        self.samples, self.background_models = self.get_all_samples()

        self.jump = jump
        self.hold = hold
        self.return_normal_seq = return_normal_seq  # for fast and slow moving
        self.tube_mask = tube_mask  # True if masking in tubes

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in natsorted(videos):
            video_name = video.split('/')[-2]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = natsorted(glob.glob(os.path.join(video, '*' + self.extension)))
            #self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        background_models = []
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in natsorted(videos):
            video_name = video.split('/')[-2]

            for i in range(len(self.videos[video_name]['frame']) - self._num_frames + 1):
                frames.append(self.videos[video_name]['frame'][i])
                # background_models.append(bg_filename)

        return frames, background_models

    def __getitem__(self, index):
        # index = 8
        video_name = self.samples[index].split('/')[-2]
        if (self.dataset == 'i_LIDS') or (self.dataset == 'shanghai' and 'training' in self.samples[index]):  # bcos my shanghai's start from 1
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2]) - 1
            #print("here")
        elif(self.dataset == 'mutton'):
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2].split('-')[-1]) - 1
        else:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        normal_batch = []
        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width, grayscale=True, normalize=False)

            if self.transform is not None:
                if self.mask_per == 0:
                    batch.append(self.transform(img_normalize(image)))
                elif self.tube_mask:
                    image = self.transform(image)
                    normal_batch.append(img_normalize(image))
                    if i == 0: # get only 1 mask in beginning!!!
                        num_patches = self._resize_height // self.psize
                        mask_generator = RandomMaskingGenerator(num_patches, self.mask_per)
                        bool_masked_pos = torch.from_numpy(mask_generator()).flatten().to(torch.bool)
                        img_patch = rearrange(image, 'c (x p1) (y p2) -> (x y) (p1 p2 c)', p1=self.psize,
                                              p2=self.psize)  # psize
                        mask = torch.ones_like(img_patch)
                        mask[bool_masked_pos] = 0
                        mask = rearrange(mask, '(x y) (p1 p2 c) -> c (x p1) (y p2)', p1=self.psize, p2=self.psize,
                                         x=num_patches, y=num_patches)

                    image = img_normalize(image * mask)
                    batch.append(image)
                else:
                    image = self.transform(image)
                    normal_batch.append(img_normalize(image))
                    num_patches = self._resize_height // self.psize
                    mask_generator = RandomMaskingGenerator(num_patches, self.mask_per)
                    bool_masked_pos = torch.from_numpy(mask_generator()).flatten().to(torch.bool)
                    img_patch = rearrange(image, 'c (x p1) (y p2) -> (x y) (p1 p2 c)', p1=self.psize, p2=self.psize)  # psize
                    mask = torch.ones_like(img_patch)
                    mask[bool_masked_pos] = 0
                    mask = rearrange(mask, '(x y) (p1 p2 c) -> c (x p1) (y p2)', p1=self.psize, p2=self.psize, x=num_patches, y=num_patches)      
                    image = img_normalize( image * mask )
                    batch.append(image)

        if self.mask_per == 0:
            return np.stack(batch, axis=1)
        else:
            return np.stack(batch, axis=1), np.stack(normal_batch, axis=1)

    def __len__(self):
        return len(self.samples)


class Reconstruction3DDataLoaderJump(Reconstruction3DDataLoader):
    def __getitem__(self, index):
        # index = 8
        video_name = self.samples[index].split('/')[-2]
        #print("reconjump : index, video ", index, video_name)
        if (self.dataset == 'i_LIDS') or (self.dataset == 'shanghai' and 'training' in self.samples[index]):  # bcos my shanghai's start from 1
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2]) - 1
            #print("here")
        elif(self.dataset == 'mutton'):
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2].split('-')[-1]) - 1
        else:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        normal_batch = []
        jump = random.choice(self.jump)

        retry = 0
        while len(self.videos[video_name]['frame']) < frame_name + (self._num_frames-1) * jump and retry < 10:
            # reselect the frame_name
            frame_name = np.random.randint(len(self.videos[video_name]['frame']))
            retry += 1

        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i*jump, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                  self._resize_width, grayscale=True, normalize=False)

            if self.transform is not None:
                batch.append(self.transform(img_normalize(image)))
                '''

                if self.mask_per == 0:
                    batch.append(self.transform(img_normalize(image)))
                else:
                    image = self.transform(image)
                    num_patches = self._resize_height // self.psize
                    mask_generator = RandomMaskingGenerator(num_patches, self.mask_per)
                    bool_masked_pos = torch.from_numpy(mask_generator()).flatten().to(torch.bool)
                    img_patch = rearrange(image, 'c (x p1) (y p2) -> (x y) (p1 p2 c)', p1=self.psize, p2=self.psize)  # psize
                    mask = torch.ones_like(img_patch)
                    mask[bool_masked_pos] = 0
                    mask = rearrange(mask, '(x y) (p1 p2 c) -> c (x p1) (y p2)', p1=self.psize, p2=self.psize, x=num_patches, y=num_patches)
                    image = img_normalize( image * mask )
                    batch.append(image)
		''' 

        if self.return_normal_seq:
            for i in range(self._num_frames):
                image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                      self._resize_width, grayscale=True, normalize=False)

                if self.transform is not None:
                    normal_batch.append(self.transform(img_normalize(image)))

                    ''' forget mask for now
                    if self.mask_per == 0:
                        normal_batch.append(self.transform(img_normalize(image)))
                    else:
                        image = self.transform(image)
                        num_patches = self._resize_height // self.psize
                        mask_generator = RandomMaskingGenerator(num_patches, self.mask_per)
                        bool_masked_pos = torch.from_numpy(mask_generator()).flatten().to(torch.bool)
                        img_patch = rearrange(image, 'c (x p1) (y p2) -> (x y) (p1 p2 c)', p1=self.psize,
                                              p2=self.psize)  # psize
                        mask = torch.ones_like(img_patch)
                        mask[bool_masked_pos] = 0
                        mask = rearrange(mask, '(x y) (p1 p2 c) -> c (x p1) (y p2)', p1=self.psize, p2=self.psize,
                                         x=num_patches, y=num_patches)
                        image = img_normalize(image * mask)
                        normal_batch.append(image)
                    '''
            return np.stack(batch, axis=1), np.stack(normal_batch, axis=1)

        else:
            return np.stack(batch, axis=1), normal_batch
