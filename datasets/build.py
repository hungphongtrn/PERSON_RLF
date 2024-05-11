from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import torch
import torchvision.transforms as T

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .vn3k import VN3K

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, "VN3K": VN3K}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    # Unpack the list of samples into separate lists for each attribute
    pids = [item['pids'] for item in batch]
    image_ids = [item['image_ids'] for item in batch]
    images = [item['images'] for item in batch]

    # Handle caption_ids and mlm_ids
    caption_input_ids = [item['caption_ids']['input_ids'] for item in batch]
    caption_attention_mask = [item['caption_ids']['attention_mask'] for item in batch]
    mlm_masked_input_ids = [item['mlm_ids']['input_ids'] for item in batch]
    mlm_masked_attention_mask = [item['mlm_ids']['attention_mask'] for item in batch]
    mlm_masked_labels = [item['mlm_labels']['input_ids'] for item in batch]

    # Stack tensors
    images = torch.stack(images)  # Assuming all images are of the same size
    caption_input_ids = torch.cat(caption_input_ids, dim=0)
    caption_attention_mask = torch.cat(caption_attention_mask, dim=0)
    mlm_masked_input_ids = torch.cat(mlm_masked_input_ids, dim=0)
    mlm_masked_attention_mask = torch.cat(mlm_masked_attention_mask, dim=0)
    mlm_masked_labels = torch.cat(mlm_masked_labels, dim=0)

    # Convert lists of pids and image_ids to tensors
    pids = torch.tensor(pids, dtype=torch.int64)
    image_ids = torch.tensor(image_ids, dtype=torch.int64)

    # Create a dictionary to return
    batched_data = {
        'pids': pids,
        'image_ids': image_ids,
        'images': images,
        'caption_ids': {
            'input_ids': caption_input_ids,
            'attention_mask': caption_attention_mask
        },
        'mlm_ids': {
            'input_ids': mlm_masked_input_ids,
            'attention_mask': mlm_masked_attention_mask
        },
        'mlm_labels': mlm_masked_labels
    }

    return batched_data


def test_image_collate(batch):
    pids = [item['pids'] for item in batch]
    images = [item['images'] for item in batch]

    images = torch.stack(images)

    pids = torch.tensor(pids, dtype=torch.int64)

    return {'pids': pids, 'images': images}


def test_text_collate(batch):
    pids = [item['pids'] for item in batch]

    caption_input_ids = [item['caption_ids']['input_ids'] for item in batch]
    caption_attention_mask = [item['caption_ids']['attention_mask'] for item in batch]

    pids = torch.tensor(pids)
    caption_input_ids = torch.cat(caption_input_ids, dim=0)
    caption_attention_mask = torch.cat(caption_attention_mask, dim=0)

    return {'pids': pids, 'caption_ids': {'input_ids': caption_input_ids, 'attention_mask': caption_attention_mask}}

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        # Create augmentations for images 
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            # MLM masking
            train_set = ImageTextMLMDataset(dataset.train,
                                            train_transforms,
                                            text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=test_image_collate)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=test_text_collate)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     collate_fn=test_image_collate)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     collate_fn=test_text_collate)
        return test_img_loader, test_txt_loader, num_classes
