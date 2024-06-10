from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from loguru import logger
import torch
import pytorch_lightning as pl

# from torch.utils.data.distributed import DistributedSampler

from datasets.bases import (
    ImageDataset, 
    TextDataset,
    ImageTextDataset,
    ImageTextMLMDataset
)
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.vn3k import VN3K
from datasets.build import build_transforms, collate
from utils.tokenizer import get_tokenizer
from utils.comm import get_world_size




class IRRADataModule(pl.LightningDataModule):
    def __init__(self, args, transforms=None):
        super().__init__()
        __factory = {'CUHK-PEDES': CUHKPEDES,
                     'ICFG-PEDES': ICFGPEDES,
                     'RSTPReid': RSTPReid,
                     "VN3K": VN3K}
        self.args = args
        self.transforms = transforms
        print(args.dataset_name)
        self.dataset = __factory[args.dataset_name](root=args.root_dir)
        self.tokenizer = get_tokenizer(args)
        self.num_classes = len(self.dataset.train_id_container)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_transforms = build_transforms(img_size=self.args.img_size,
                                                aug=self.args.img_aug,
                                                is_train=True)
            val_transforms = build_transforms(img_size=self.args.img_size,
                                              is_train=False)

            if self.args.MLM:
                self.train_set = ImageTextMLMDataset(self.tokenizer,
                                                     self.dataset.train,
                                                     train_transforms,
                                                     text_length=self.args.text_length)
            else:
                self.train_set = ImageTextDataset(self.tokenizer, 
                                                  self.dataset.train,
                                                  train_transforms,
                                                  text_length=self.args.text_length)

            ds = self.dataset.val
            self.val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                            val_transforms)
            self.val_txt_set = TextDataset(self.tokenizer,
                                           ds['caption_pids'],
                                           ds['captions'],
                                           text_length=self.args.text_length)

        if stage == 'test' or stage is None:
            if self.transforms:
                test_transforms = self.transforms
            else:
                test_transforms = build_transforms(img_size=self.args.img_size,
                                                   is_train=False)

            ds = self.dataset.test
            self.test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                             test_transforms)
            self.test_txt_set = TextDataset(self.tokenizer,
                                            ds['caption_pids'],
                                            ds['captions'],
                                            text_length=self.args.text_length)

    def train_dataloader(self):
        if self.args.sampler == 'identity':
            if self.args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = self.args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    self.dataset.train, self.args.batch_size, self.args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                return DataLoader(self.train_set,
                                  batch_sampler=batch_sampler,
                                  num_workers=self.args.num_workers,
                                  collate_fn=collate)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {self.args.batch_size}, id: {self.args.batch_size // self.args.num_instance}, instance: {self.args.num_instance}'
                )
                return DataLoader(self.train_set,
                                  batch_size=self.args.batch_size,
                                  sampler=RandomIdentitySampler(
                                      self.dataset.train, self.args.batch_size,
                                      self.args.num_instance),
                                  num_workers=self.args.num_workers,
                                  collate_fn=collate)
        elif self.args.sampler == 'random':
            logger.info('using random sampler')
            return DataLoader(self.train_set,
                              batch_size=self.args.batch_size,
                              shuffle=True,
                              num_workers=self.args.num_workers,
                              collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(self.args.sampler))
            raise ValueError('Unsupported sampler type')

    def val_dataloader(self):
        val_img_loader = DataLoader(self.val_img_set,
                                    batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    collate_fn=collate)
        val_txt_loader = DataLoader(self.val_txt_set,
                                    batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    collate_fn=collate)
        
        return [val_img_loader, val_txt_loader]

    def test_dataloader(self):
        test_img_loader = DataLoader(self.test_img_set,
                                     batch_size=self.args.test_batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     collate_fn=collate)
        test_txt_loader = DataLoader(self.test_txt_set,
                                     batch_size=self.args.test_batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     collate_fn=collate)
        return [test_img_loader, test_txt_loader]
