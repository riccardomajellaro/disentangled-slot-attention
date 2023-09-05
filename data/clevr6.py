# THIS SCRIPT MUST BE RUN ONLY AFTER clevr.py IN ORDER TO FILTER IT

from os.path import join
import numpy as np
import argparse
import os


def main(clevr_path, clevr6_path):
    for split in ["train", "test"]:
        for m in os.listdir(join(clevr_path, split, "masks")):
            masks = np.load(join(clevr_path, split, "masks", m), allow_pickle=True)
            if np.sum(masks.sum((1,2,3))>0) <= 7:
                image = np.load(join(clevr_path, split, "images", m.replace("mask","image")), allow_pickle=True)
                np.save(join(clevr6_path, split, "masks", m), masks)
                np.save(join(clevr6_path, split, "images", m.replace("mask","image")), image)

        shapes = np.load(join(clevr_path, split, "shapes.npy"), allow_pickle=True)
        colors = np.load(join(clevr_path, split, "colors.npy"), allow_pickle=True)
        materials = np.load(join(clevr_path, split, "materials.npy"), allow_pickle=True)
        shapes_6, colors_6, materials_6 = [], [], []
        for i in range(shapes.shape[0]):
            if (shapes[i] > 0).sum() < 7:
                shapes_6.append(shapes[i][:7])
                colors_6.append(colors[i][:7])
                materials_6.append(materials[i][:7])
            else:
                shapes_6.append(np.zeros(7))
                colors_6.append(np.zeros(7))
                materials_6.append(np.zeros(7))
        shapes_6 = np.stack(shapes_6)
        colors_6 = np.stack(colors_6)
        materials_6 = np.stack(materials_6)
        np.save(join(clevr6_path, split, "shapes"), shapes_6)
        np.save(join(clevr6_path, split, "colors"), colors_6)
        np.save(join(clevr6_path, split, "materials"), materials_6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevr_path", default=None, type=str, help="Path of existing CLEVR dataset", required=True)
    parser.add_argument("--clevr6_path", default=None, type=str, help="Path where CLEVR6 should be stored", required=True)
    args = parser.parse_args()
    args = vars(args)
    main(args["clevr_path"], args["clevr6_path"])