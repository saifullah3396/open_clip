from iitcdip import TRAIN_FILES, VAL_FILES, IITCDIPDatasetPreprocessor
import tqdm
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torch
import numpy as np
import argparse


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
        dataset, batch_size=100, num_workers=8, shuffle=False, collate_fn=lambda x: x
    )

    # generate output filename
    if args.split == "train":
        output_file = f"preprocess_{args.split}_0_{TRAIN_FILES}"
    else:
        output_file = f"preprocess_{args.split}_{TRAIN_FILES}_{TRAIN_FILES+VAL_FILES}"

    total = 0
    indices_list = []
    labels_list = []
    pbar = tqdm.tqdm(dataloader)
    for data in pbar:
        with torch.no_grad():
            index = torch.tensor([d[0] for d in data])
            success = torch.tensor([d[1] for d in data])
            labels = [d[2] for d in data]
            indices_list += index[success == 1].tolist()
            labels_list += np.array(labels)[success == 1].tolist()
            total += len(index)
            pbar.set_postfix({"total": total, "indices": len(indices_list)})
    data = {"indices": indices_list, "labels": labels_list}
    with open(f"{output_file}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
