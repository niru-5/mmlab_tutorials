import os
import pandas as pd
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_200_2011_dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=42)
    return parser.parse_args()


def main(args):
    CUB_200_2011_dataset = args.cub_200_2011_dataset
    random_seed = args.seed

    label_map = {}
    with open(os.path.join(CUB_200_2011_dataset, "classes.txt"), "r") as f:
        class_names = f.readlines()
        for idx, class_name in enumerate(class_names):
            tokens = class_name.strip().split()
            label_map[tokens[1]] = tokens[0]



    image_id_map = {}
    with open(os.path.join(CUB_200_2011_dataset, "train_test_split.txt"), "r") as f:
        train_test_split = f.readlines()
        for line in train_test_split:
            tokens = line.strip().split()
            image_id_map[tokens[0]] = tokens[1]


    image_data = defaultdict(list)
    with open(os.path.join(CUB_200_2011_dataset, "images.txt"), "r") as f:
        images = f.readlines()
        for image in images:
            tokens = image.strip().split()
            # image_map[tokens[0]] = tokens[1].split("/")[0]
            image_data['img_id'].append(int(tokens[0]))
            image_data['img_path'].append(tokens[1])
            image_data['img_class'].append(tokens[1].split("/")[0])
            image_data['img_class_id'].append(label_map[tokens[1].split("/")[0]])
            image_data['img_split'].append(int(image_id_map[tokens[0]]))

    image_df = pd.DataFrame(image_data)
    train_df = image_df[image_df['img_split'] == 1]
    validation_df = train_df.groupby(['img_class']).sample(frac=0.1, random_state=random_seed)
    train_df = train_df.drop(validation_df.index)
    test_df = image_df[image_df['img_split'] == 0]

    train_df.to_csv(os.path.join(CUB_200_2011_dataset, "train_df.csv"), index=False)
    validation_df.to_csv(os.path.join(CUB_200_2011_dataset, "validation_df.csv"), index=False)
    test_df.to_csv(os.path.join(CUB_200_2011_dataset, "test_df.csv"), index=False)  

    print (f"train_df.shape: {train_df.shape[0]}")
    print (f"validation_df.shape: {validation_df.shape[0]}")
    print (f"test_df.shape: {test_df.shape[0]}")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)