from __future__ import annotations

import io
import pickle
import PIL
from datadings.reader import MsgpackReader
import torch
from torch.utils.data import Dataset
import numpy as np
from training.datasets.constants import MSGPACK_FILES
from PIL import Image
import xml.dom.minidom
from urllib.request import urlopen
import pickle

TRAIN_FILES = 125
TRAIN_MSGPACK_FILES = MSGPACK_FILES[:TRAIN_FILES]

VAL_FILES = 2
VAL_MSGPACK_FILES = MSGPACK_FILES[TRAIN_FILES : TRAIN_FILES + VAL_FILES]
RVLCDIP_PATH = './rvlcdip_labels/'

class GrayScaleToRGB:
    """
    Converts a gray-scale torch image to rgb image.
    """

    def __call__(self, image: torch.Tensor):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


class IITCDIPDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, tokenizer=None):
        self.data_path = data_path
        self.msgpack_readers = []
        files = TRAIN_MSGPACK_FILES if split == "train" else VAL_MSGPACK_FILES
        for file in files:
            try:
                self.msgpack_readers.append(MsgpackReader(data_path + file))
            except:
                pass

        self.total_lens = []
        self.total_len = 0
        self.map_indices = []
        for idx, msgpack_reader in enumerate(self.msgpack_readers):
            self.map_indices += [idx] * len(msgpack_reader)
            self.total_lens.append(self.total_len)
            self.total_len += len(msgpack_reader)
        self.transform = transform
        self.tokenizer = tokenizer

        # now load preprocess data
        if split == "train":
            preprocess_data_file = f"{data_path}/preprocess_data/preprocess_{split}_0_{TRAIN_FILES}.pkl"
        else:
            preprocess_data_file = (
                f"{data_path}/preprocess_data/preprocess_{split}_{TRAIN_FILES}_{TRAIN_FILES+VAL_FILES}.pkl"
            )

        with open(preprocess_data_file, "rb") as f:
            preprocess_data = pickle.load(f)
            self.labels = preprocess_data["labels"]
            self.clean_indices = preprocess_data["indices"]

    def get_sample(self, index):
        msgpack_index = self.map_indices[index]
        reader = self.msgpack_readers[msgpack_index]
        return reader[index - self.total_lens[msgpack_index]]

    def __getitem__(self, index):
        # get clean index
        labels = self.labels[index]
        index = self.clean_indices[index]

        # load sample
        sample = self.get_sample(index)

        # get sample caption
        sample["caption"] = f"A document of following types: {labels.lower()}."

        # decode image if required
        image_load_map = {
            "image": "image_file_path",
        }

        for image_key, path_key in image_load_map.items():
            if image_key in sample and isinstance(sample[image_key], (bytes, str)):
                sample[image_key] = PIL.Image.fromarray(
                    np.array(PIL.Image.open(io.BytesIO(sample[image_key]))) * 255.0
                ).convert("RGB")
            elif image_key in sample and isinstance(sample[image_key], (np.ndarray)):
                sample[image_key] = PIL.Image.fromarray(sample[image_key]).convert(
                    "RGB"
                )

        # apply transform
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])

        print('sample', sample['caption'])
        return sample["image"], self.tokenizer(sample["caption"])[0]

    def __len__(self):
        return len(self.clean_indices)


class IITCDIPDatasetPreprocessor(IITCDIPDataset):
    def __init__(self, data_path, split="train", transform=None, tokenizer=None):
        super().__init__(data_path, split, transform, tokenizer)
        self.rvlcdip_paths = []
        for split in ['train', 'test', 'val']:
            label_file = RVLCDIP_PATH + split + '.txt'
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    self.rvlcdip_paths.append(line.split(' ')[0].strip())
        self.rvlcdip_overlap_count = 0
        
    def __getitem__(self, index):
        try:
            # load sample
            sample = self.get_sample(index)
            
            # see if this sample overlaps with RVL-CDIP
            if sample['image_file_path'] in self.rvlcdip_paths:
                print("path in rvldcip found", sample['image_file_path'])   
                print('self.rvlcdip_overlap_count', self.rvlcdip_overlap_count)
                self.rvlcdip_overlap_count+=1
                return index, 0, ""

            if "xml_file" not in sample:
                return index, 0, ""
            else:
                xml_str = str(sample["xml_file"])
                start = xml_str.find("<dt>")
                end = xml_str.find("</dt>")
                labels = xml_str[start + 4 : end]
                if len(labels) > 0:
                    return index, 1, labels
                else:
                    return index, 0, ""
        except KeyboardInterrupt:
            exit(1)
        except Exception:
            return index, 0, ""

    def __len__(self):
        return self.total_len
