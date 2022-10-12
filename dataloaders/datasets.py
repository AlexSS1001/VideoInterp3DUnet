import argparse
import kornia.augmentation as K_augs
import kornia.filters as K_filters
import h5py
import numpy as np
import os
import torch
from PIL import Image
from torch import true_divide
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 933120000
import warnings
warnings.filterwarnings("ignore")


# split into train and valid datasets
class DatasetTrain(Dataset):
    def __init__(self, args):
        super().__init__()
        self.train_db_path = args['train_h5_db_path']
        self.valid_db_path = args['valid_h5_db_path']

        self.check_databases(args)
        self.shard_list = os.listdir(self.train_db_path)
        self.shard_list.sort()
        self.seq_list = []
        self.total_samples = 0
        for shard in self.shard_list:
            shard_path = os.path.join(self.train_db_path, shard)
            with h5py.File(shard_path, 'r') as fh5:
                keys = list(fh5.keys())
                self.seq_list.append({shard: keys})
        
        # augmentations
        self.aug_list = K_augs.AugmentationSequential(
            K_augs.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
            K_augs.RandomHorizontalFlip(p=0.5),
            K_augs.RandomVerticalFlip(p=0.5),
            K_augs.RandomCrop(size=(args['crop_h'], args['crop_w']), p=1.0),
            data_keys=["input"],
            same_on_batch=True,
            random_apply=False)

    def check_databases(self, args):
        # check training database
        if os.path.isdir(self.train_db_path) and len(os.listdir(self.train_db_path)) > 0:
            pass
        else:
            if not os.path.isdir(self.train_db_path):
                os.makedirs(self.train_db_path)
            else:
                os.removedirs(self.train_db_path)
                os.makedirs(self.train_db_path)
                print('Generating training database')
                DatasetTrain.GenerateVimeoHDF5Dataset(db_path =args['db_path'],
                                                                                    seq_list  =args['train_list'],
                                                                                    shards   =args['shards'],
                                                                                    dst_path=args['train_h5_db_path'])
        # check validation database
        if os.path.isdir(self.valid_db_path) and len(os.listdir(self.valid_db_path)) > 0:
            pass
        else:
            if not os.path.isdir(self.valid_db_path):
                os.makedirs(self.valid_db_path)
            else:
                os.removedirs(self.valid_db_path)
                os.makedirs(self.valid_db_path)
                print('Generating training database')
                DatasetTrain.GenerateVimeoHDF5Dataset(db_path  =args['db_path'],
                                                                                    seq_list   =args['valid_list'],
                                                                                    shards    =1,
                                                                                    dst_path=args['valid_h5_db_path'])
    
    @staticmethod
    def GenerateVimeoHDF5Dataset(db_path, seq_list, shards, dst_path):
        if os.path.isdir(db_path):
            if not os.path.isdir(dst_path):
                os.makedirs(dst_path)

            seq_list = os.path.join(db_path, seq_list)
            if os.path.isfile(seq_list):
                with open(seq_list, 'r') as f_sequences:
                    seq_list = f_sequences.readlines()

                    # split list into evenly sized shards
                    # last shards may have a smaller length
                    divider = shards * (len(seq_list)//shards)
                    seq_list_main    = seq_list[0:divider]
                    seq_list_remain = np.array(seq_list[divider:-1])
                    seq_shards = np.array_split(seq_list_main, shards)
                    seq_shards.append(seq_list_remain)

                    for seq_shard, shard_id, _, in zip(seq_shards, range(shards), tqdm(seq_shards)):
                        with h5py.File(os.path.join(dst_path, 'shard_%5.5d.h5' % shard_id), 'w') as fh5:
                            for seq, seq_id, _ in zip(seq_shard, range(len(seq_shard)), tqdm(seq_shard)):
                                seq = seq.rstrip('\r\n')
                                seq_path = os.path.join(db_path, 'sequences', seq)
                                images = os.listdir(seq_path)
                                img_list = []
                                
                                for img_name in images:
                                    img_list.append(Image.open(os.path.join(seq_path, img_name)).convert('RGB'))
                                # write sequence to hdf5
                                sample_w, sample_h = img_list[0].size
                                sample_name = 'septuplet_%5.5d' % seq_id

                                fh5.create_dataset(sample_name, (len(img_list), sample_h, sample_w, 3), np.uint8)

                                for img, j in zip(img_list, range(len(img_list))):
                                    fh5[sample_name][j] = img
            else:
                print('Missing file liste: {}'.format(seq_list))
        else:
            print('Invalid database path: {}'.format(db_path))
    
    @staticmethod
    def ReadVimeoHDF5(file_path, key):
        with h5py.File(file_path, 'r') as fh5:
            sample = np.transpose((fh5['septuplet_%5.5d' % key][:]), (0,3,1,2) )
        return torch.from_numpy(sample)/255.0
    
    def __len__(self):
        self.db_len = 0
        # get shard length based on first shard
        key = list(self.seq_list[0].keys())[0]
        self.shard_len = len(self.seq_list[0][key])

        for item in self.seq_list:
            key = list(item.keys())[0]
            self.db_len += len(item[key])
        return self.db_len

    def __getitem__(self, index):
        shard_id = index // self.shard_len
        if 0 == shard_id:
            sample_id = index
        else:
            sample_id = index % (shard_id * self.shard_len)

        sample = self.ReadVimeoHDF5(file_path=os.path.join(self.train_db_path, self.shard_list[shard_id]), key = sample_id)
        sample = self.aug_list(sample)

        return sample


# split into train and valid datasets
class DatasetValid(Dataset):
    def __init__(self, args):
        super().__init__()
        self.valid_db_path = args['valid_h5_db_path']

        self.shard_list = os.listdir(self.valid_db_path)
        self.shard_list.sort()
        self.seq_list = []
        self.total_samples = 0
        for shard in self.shard_list:
            shard_path = os.path.join(self.valid_db_path, shard)
            with h5py.File(shard_path, 'r') as fh5:
                keys = list(fh5.keys())
                self.seq_list.append({shard: keys})
    
    @staticmethod
    def ReadVimeoHDF5(file_path, key):
        with h5py.File(file_path, 'r') as fh5:
            sample = np.transpose((fh5['septuplet_%5.5d' % key][:]), (0,3,1,2) )
        return torch.from_numpy(sample)/255.0
        
    def __len__(self):
        self.db_len = 0
        # get shard length based on first shard
        key = list(self.seq_list[0].keys())[0]
        self.shard_len = len(self.seq_list[0][key])

        for item in self.seq_list:
            key = list(item.keys())[0]
            self.db_len += len(item[key])
        return self.db_len

    def __getitem__(self, index):
        shard_id = index // self.shard_len
        if 0 == shard_id:
            sample_id = index
        else:
            sample_id = index % (shard_id * self.shard_len)

        sample = self.ReadVimeoHDF5(file_path=os.path.join(self.valid_db_path, self.shard_list[shard_id]), key = sample_id)

        return sample


class DatasetInference(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.test_db_path = args.test_db_path
        self.img_list = os.listdir(args.test_db_path)
        self.img_list.sort()
        self.img_formats = ['.jpg', '.png', '.tiff']
        self.img_width = args.img_width
        self.img_height = args.img_height
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.test_db_path, self.img_list[idx])
        img_ext = os.path.splitext(img_path)[-1]
        if img_ext in self.img_formats and os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            if -1 == self.img_width or -1 == self.img_height:
                blank_img_w = 256
                blank_img_h = 256
            else:
                blank_img_w = self.img_width
                blank_img_h = self.img_height
            print('Invalid image: {}'.format(img_path))
            img = Image.new('RGB', (blank_img_w, blank_img_h))
        img = img.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        return ToTensor()(img)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-src-path', type=str, required=True, default=r'/mnt/f/Projects/video_interp_pl/DB/vimeo_septuplet')
    parser.add_argument('--db-h5-path', type=str, required=True, default=r'/mnt/d/vimeo_septuplet_h5')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seq_list = 'sep_trainlist.txt'
    DatasetTrain.GenerateVimeoHDF5Dataset(db_path=args.db_src_path, seq_list=seq_list, shards=150, dst_path=args.db_h5_path)