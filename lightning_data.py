import logging
import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.augmentation.transform import (
    build_image_augmentation_pool,
    build_text_augmentation_pool,
    get_self_supervised_augmentation,
)
from datasets.bases import (
    ImageDataset,
    ImageTextDataset,
    ImageTextMLMDataset,
    TextDataset,
)
from datasets.build import collate
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from datasets.vn3k_mixed import VN3K_MIXED
from datasets.vn3k_vi import VN3K_VI
from datasets.vn3k_en import VN3K_EN
from utils.comm import get_world_size
from utils.tokenizer_utils import get_tokenizer

# from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
# Filter UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class TBPSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        __factory = {
            "CUHK-PEDES": CUHKPEDES,
            "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid,
            "VN3K_EN": VN3K_EN,
            "VN3K_VI": VN3K_VI,
            "VN3K_MIXED": VN3K_MIXED,
        }
        self.config = config
        print(config.experiment.dataset_name)
        self.dataset = __factory[config.experiment.dataset_name](
            root=config.dataset_root_dir
        )
        self.tokenizer = get_tokenizer(config.tokenizer)
        self.num_classes = len(self.dataset.train_id_container)
        self.collate_fn = collate
        # Set up transforms
        self._prepare_transforms()

    def _prepare_transforms(self):
        if self.config.experiment.aug.img.pop("ss_aug"):
            self.ss_image_transform = get_self_supervised_augmentation(
                self.config.experiment.aug.img.size
            )
        else:
            self.ss_image_transform = None

        # Image augmentation can't be None; at minium, contain Resize, ToTensor, and Normalize
        self.image_transform = build_image_augmentation_pool(
            **self.config.experiment.aug.img, is_train=True
        )
        self.test_image_transforms = build_image_augmentation_pool(
            **self.config.experiment.aug.img, is_train=False
        )

        # Text augmentaion can be None, in which case, no augmentation is applied
        self.text_transform = build_text_augmentation_pool(
            **self.config.experiment.aug.text, is_train=True
        )
        self.test_text_transforms = build_text_augmentation_pool(
            **self.config.experiment.aug.text, is_train=False
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.config.experiment.MLM:
                self.train_set = ImageTextMLMDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    image_transform=self.image_transform,
                    ss_image_transform=self.ss_image_transform,
                    text_transform=self.text_transform,
                )
            else:
                self.train_set = ImageTextDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    image_transform=self.image_transform,
                    ss_image_transform=self.ss_image_transform,
                    text_transform=self.text_transform,
                )

            if self.dataset.val:
                logger.info("Validation set is available")
                self.val_img_set = ImageDataset(
                    dataset=self.dataset.val,
                    image_transform=self.test_image_transforms,
                )
                self.val_txt_set = TextDataset(
                    dataset=self.dataset.val,
                    tokenizer=self.tokenizer,
                    text_transform=self.test_text_transforms,
                )

        if stage == "test" or stage is None:
            self.test_img_set = ImageDataset(
                dataset=self.dataset.test,
                image_transform=self.test_image_transforms,
            )
            self.test_txt_set = TextDataset(
                dataset=self.dataset.test,
                tokenizer=self.tokenizer,
                text_transform=self.test_text_transforms,
            )

    def train_dataloader(self):
        if self.config.experiment.sampler == "identity":
            if self.config.distributed:
                logger.info("using ddp random identity sampler")
                logger.info("DISTRIBUTED TRAIN START")
                mini_batch_size = self.config.experiment.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    self.dataset.train,
                    self.config.experiment.batch_size,
                    self.config.experiment.num_instance,
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True
                )
                return DataLoader(
                    self.train_set,
                    batch_sampler=batch_sampler,
                    num_workers=self.config.experiment.num_workers,
                    collate_fn=self.collate_fn,
                )
            else:
                logger.info(
                    f"using random identity sampler: batch_size: {self.config.experiment.batch_size}, id: {self.config.experiment.batch_size // self.config.experiment.num_instance}, instance: {self.config.experiment.num_instance}"
                )
                return DataLoader(
                    self.train_set,
                    batch_size=self.config.experiment.batch_size,
                    sampler=RandomIdentitySampler(
                        self.dataset.train,
                        self.config.experiment.batch_size,
                        self.config.experiment.num_instance,
                    ),
                    num_workers=self.config.experiment.num_workers,
                    collate_fn=self.collate_fn,
                )
        elif self.config.experiment.sampler == "random":
            logger.info("using random sampler")
            return DataLoader(
                self.train_set,
                batch_size=self.config.experiment.batch_size,
                shuffle=True,
                num_workers=self.config.experiment.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            logger.error(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    self.config.experiment.sampler
                )
            )
            raise ValueError("Unsupported sampler type")

    # def val_dataloader(self):
    #     val_img_loader = DataLoader(
    #         self.val_img_set,
    #         batch_size=self.config.experiment.batch_size,
    #         shuffle=False,
    #         num_workers=self.config.experiment.num_workers,
    #         collate_fn=padded_collate,
    #     )
    #     val_txt_loader = DataLoader(
    #         self.val_txt_set,
    #         batch_size=self.config.experiment.batch_size,
    #         shuffle=False,
    #         num_workers=self.config.experiment.num_workers,
    #         collate_fn=padded_collate,
    #     )

    #     return [val_img_loader, val_txt_loader]

    def test_dataloader(self):
        test_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.experiment.test_batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            collate_fn=self.collate_fn,
        )
        test_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.experiment.test_batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            collate_fn=self.collate_fn,
        )
        return [test_img_loader, test_txt_loader]
