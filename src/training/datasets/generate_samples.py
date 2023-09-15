from iitcdip import TRAIN_FILES, VAL_FILES, IITCDIPDatasetPreprocessor
import tqdm
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torch
import numpy as np
import argparse
from .openclip_transform import image_transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="/ds/documents/IIT-CDIP/",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--split", default="train", help="Which dataset split to preprocess."
    )

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args

    # load dataset
    dataset = IITCDIPDatasetPreprocessor(
        data_path=args.data_path,
        split=args.split,
    )

    # create dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x
    )

    # get preprocess transform
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )
    
    for data in tqdm.tqdm(dataloader):
        for 
        print(data.keys())
        break 



if __name__ == "__main__":
    main()
