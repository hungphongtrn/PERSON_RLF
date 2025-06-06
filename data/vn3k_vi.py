import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class VN3K_VI(BaseDataset):
    """
    # identities: 6302
    """

    dataset_dir = "VN3K"

    def __init__(self, root="", verbose=True, seed=42):
        super(VN3K_VI, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, "imgs/")

        self.anno_path = op.join(self.dataset_dir, "data_captions.json")
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(
            self.anno_path
        )

        self.train, self.train_id_container = self._process_anno(
            self.train_annos, training=True
        )
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> VN3K_VI Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno["split"] == "train":
                train_annos.append(anno)
            elif anno["split"] == "test":
                test_annos.append(anno)
            elif anno["split"] == "validate":
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno["id"]) - 1
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno["file_path"])
                captions = anno["captions"]  # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}  # type: ignore
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno["id"]) - 1
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno["file_path"])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno["captions"]  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            # Return None if the dataset is empty
            if not image_pids and not caption_pids and not img_paths and not captions:
                return None, None
            dataset = {
                "image_pids": image_pids,  # type: ignore
                "img_paths": img_paths,  # type: ignore
                "caption_pids": caption_pids,  # type: ignore
                "captions": captions,  # type: ignore
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
