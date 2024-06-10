from typing import Any, Optional

from loguru import logger
from prettytable import PrettyTable
from torch.utils.data import Dataset
import torch

from utils.iotools import read_image


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logger
    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=187, truncate=True):
    inputs = tokenizer(caption,
                       max_length=text_length,
                       padding="max_length",
                       truncation=truncate,
                       return_tensors='pt',
                       return_attention_mask=True)
    return inputs


class ImageTextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset,
                 transform=None,
                 text_length: int = 64,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate).data

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_input': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return {
            "pids": pid,
            "images": img,
        }


class TextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 caption_pids,
                 captions,
                 text_length: int = 64,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate).data

        return {
            "pids": pid,
            "caption_input": caption,
        }


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset,
                 transform=None,
                 text_length: int = 64,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.mlm_probability = 0.15

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate).data

        inputs, labels = self.torch_mask_tokens(inputs=caption_tokens)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_input': caption_tokens,
            'mlm_inputs': inputs,
            'mlm_labels': labels
        }

        return ret

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = {
            "input_ids": inputs["input_ids"].clone(),
            "attention_mask": inputs["attention_mask"].clone(),
        }

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels['input_ids'].shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels['input_ids'].tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels['input_ids'][~masked_indices] = self.tokenizer.pad_token_id  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels['input_ids'].shape, 0.8)).bool() & masked_indices
        inputs['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels['input_ids'].shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels['input_ids'].shape, dtype=torch.long)
        inputs['input_ids'][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels['input_ids']
