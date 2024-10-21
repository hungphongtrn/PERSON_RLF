import logging
import warnings

import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, random_split

from datasets.augmentation.transform import build_image_aug_pool, build_text_aug_pool
from datasets.bases import (
    ImageDataset,
    ImageTextDataset,
    ImageTextMLMDataset,
    TextDataset,
)
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
        print(config.dataset.dataset_name)
        self.dataset = __factory[config.dataset.dataset_name](
            root=config.dataset_root_dir
        )
        self.tokenizer = get_tokenizer(config.tokenizer)
        self.num_classes = len(self.dataset.train_id_container)
        # Set up transforms
        self._prepare_augmentation_pool()

    def _prepare_augmentation_pool(self):
        self.image_aug_pool = build_image_aug_pool(self.config.aug.img.augment_cfg)
        self.text_aug_pool = build_text_aug_pool(self.config.aug.text.augment_cfg)
        self.image_random_k = self.config.aug.image_random_k
        self.text_random_k = self.config.aug.text_random_k
        if self.config.loss.SS:
            self.ss_aug = True
        else:
            self.ss_aug = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.config.loss.MLM:
                self.train_set = ImageTextMLMDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    ss_aug=self.ss_aug,
                    image_augmentation_pool=self.image_aug_pool,
                    text_augmentation_pool=self.text_aug_pool,
                    image_random_k=self.image_random_k,
                    text_random_k=self.text_random_k,
                    truncate=True,
                    image_size=self.config.aug.img.size,
                    is_train=True,
                )
            else:
                self.train_set = ImageTextDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    ss_aug=self.ss_aug,
                    image_augmentation_pool=self.image_aug_pool,
                    text_augmentation_pool=self.text_aug_pool,
                    image_random_k=self.image_random_k,
                    text_random_k=self.text_random_k,
                    truncate=True,
                    image_size=self.config.aug.img.size,
                    is_train=True,
                )

            if self.config.dataset.proportion:
                self.train_set, _ = random_split(
                    self.train_set,
                    [
                        self.config.dataset.proportion,
                        1.0 - self.config.dataset.proportion,
                    ],
                )
                logger.info(
                    f"Using {self.config.dataset.proportion} of the training set"
                )

            if self.dataset.val:
                logger.info("Validation set is available")
                self.val_img_set = ImageDataset(
                    dataset=self.dataset.val,
                    is_train=False,
                )
                self.val_txt_set = TextDataset(
                    dataset=self.dataset.val,
                    tokenizer=self.tokenizer,
                    is_train=False,
                )

        if stage == "test" or stage is None:
            self.test_img_set = ImageDataset(
                dataset=self.dataset.test,
            )
            self.test_txt_set = TextDataset(
                dataset=self.dataset.test,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        if self.config.dataset.sampler == "identity":
            if self.config.distributed:
                logger.info("using ddp random identity sampler")
                logger.info("DISTRIBUTED TRAIN START")
                mini_batch_size = self.config.dataset.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    self.dataset.train,
                    self.config.dataset.batch_size,
                    self.config.dataset.num_instance,
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True
                )
                return DataLoader(
                    self.train_set,
                    batch_sampler=batch_sampler,
                    num_workers=self.config.dataset.num_workers,
                    # collate_fn=self.collate_fn,
                    drop_last=False,
                )
            else:
                logger.info(
                    f"using random identity sampler: batch_size: {self.config.dataset.batch_size}, id: {self.config.dataset.batch_size // self.config.dataset.num_instance}, instance: {self.config.dataset.num_instance}"
                )
                return DataLoader(
                    self.train_set,
                    batch_size=self.config.dataset.batch_size,
                    sampler=RandomIdentitySampler(
                        self.dataset.train,
                        self.config.dataset.batch_size,
                        self.config.dataset.num_instance,
                    ),
                    num_workers=self.config.dataset.num_workers,
                    # collate_fn=self.collate_fn,
                    drop_last=False,
                )
        elif self.config.dataset.sampler == "random":
            logger.info("using random sampler")
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset.batch_size,
                shuffle=True,
                num_workers=self.config.dataset.num_workers,
                # collate_fn=self.collate_fn,
                drop_last=True,
            )
        else:
            logger.error(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    self.config.dataset.sampler
                )
            )
            raise ValueError("Unsupported sampler type")

    def val_dataloader(self):
        val_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        val_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_val = {
            "img": val_img_loader,
            "txt": val_txt_loader,
        }
        combined_loader = CombinedLoader(combined_val, mode="max_size")
        return combined_loader

    def test_dataloader(self):
        test_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        test_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_test = {
            "img": test_img_loader,
            "txt": test_txt_loader,
        }
        combined_loader = CombinedLoader(combined_test, mode="sequential")
        return combined_loader
