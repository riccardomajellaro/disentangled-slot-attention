from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted
import numpy as np
import torch
import os


class Dataset(Dataset):
    def __init__(self, dataset, path, split, noise=False, crop=False, resize=False, proppred=False):
        super(Dataset, self).__init__()
        assert dataset in ["tetrominoes", "multidsprites", "clevr"]
        assert split in ["train", "val", "test"]
        self.dataset = dataset
        self.split = split
        self.noise = noise
        self.crop = crop
        self.resize = resize
        self.proppred = proppred
        self.img_transform = transforms.Compose([transforms.ToTensor()])

        getattr(self, dataset + "_setup")(path)


    def tetrominoes_setup(self, path):
        self.root_dir = os.path.join(path, "train" if self.split == "val" else self.split)
        self.files = natsorted(os.listdir(os.path.join(self.root_dir, "images")))

        if self.proppred:
            if self.split == "train":
                self.files = self.files[:-1000]
            elif self.split == "val":
                self.files = self.files[-1000:]
            self.shapes = np.load(os.path.join(self.root_dir, "shapes.npy"), allow_pickle=True)
            self.colors = np.load(os.path.join(self.root_dir, "colors.npy"), allow_pickle=True)

        if self.crop == "center":
            self.crop_size = ((3, 32), (3, 32))
        if self.resize:
            self.res_size = (35, 35)

    def tetrominoes_getsample(self, index, train_split_thresh=60000):
        image_name = self.files[index]
        image = np.load(os.path.join(self.root_dir, "images", image_name), allow_pickle=True) / 255.
        image = image.astype(np.float32)

        masks_name = image_name.replace("image", "mask")
        masks = np.load(os.path.join(self.root_dir, "masks", masks_name), allow_pickle=True)

        if self.proppred:
            features_index = int(image_name.split("_")[-1].split(".")[0])
            if self.split == "test": features_index -= train_split_thresh
            features = {
                "shape": self.shapes[features_index],
                "color": self.colors[features_index]
            }
        else:
            features = None

        return image, masks, features
    
    def multidsprites_setup(self, path):
        self.tetrominoes_setup(path)

        if self.crop == "center":
            self.crop_size = ((11, 53), (11, 53))
        if self.resize:
            self.res_size = (64, 64)

    def multidsprites_getsample(self, index):
        return self.tetrominoes_getsample(index)

    def clevr_setup(self, path):
        self.tetrominoes_setup(path)

        if self.proppred:
            self.materials = np.load(os.path.join(self.root_dir, "materials.npy"), allow_pickle=True)

        if self.crop == "center":
            self.crop_size = ((29, 221), (64, 256))
        if self.resize:
            self.res_size = (128, 128)

    def clevr_getsample(self, index):
        image, masks, features = self.tetrominoes_getsample(index, train_split_thresh=70000)

        if self.proppred:
            image_name = self.files[index]
            features_index = int(image_name.split("_")[-1].split(".")[0])
            if self.split == "test": features_index -= 70000
            features["material"] = self.materials[features_index]

        return image, masks, features

    def __getitem__(self, index):
        sample = {}

        # load image and masks
        image, masks, features = getattr(self, self.dataset + "_getsample")(index)

        if self.proppred:
            for key, value in features.items():
                sample[key] = value
        
        # add plain noise
        if self.noise:
            image = (image + (np.random.randint(0,3)>0)*0.3*np.random.rand(1, 1, 3)).clip(0,1).astype(image.dtype)

        # transform images
        image = self.img_transform(image)
        if self.crop == "center":
            image = image[:, self.crop_size[0][0]:self.crop_size[0][1], self.crop_size[1][0]:self.crop_size[1][1]]
        elif self.crop == "random":
            if np.random.randint(0,2) == 0:
                crop_size = int(((np.random.rand() * 0.4) + 0.6) * min(image.shape[1:]))
            else: crop_size = min(image.shape[1:])
            crop_i, crop_j, _, _ = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
            image = F.crop(image, crop_i, crop_j, crop_size, crop_size)
        if self.resize:
            image = F.resize(image, self.res_size, antialias=None)
        sample["image"] = image

        # transform masks
        if masks is not None:
            masks = [F.to_tensor(mask) for mask in masks]
            if self.crop == "center":
                masks = [mask[:, self.crop_size[0][0]:self.crop_size[0][1], self.crop_size[1][0]:self.crop_size[1][1]] for mask in masks]
            elif self.crop == "random":
                masks = [F.crop(mask, crop_i, crop_j, crop_size, crop_size) for mask in masks]
            if self.resize:
                masks = [F.resize(mask, self.res_size, antialias=None) for mask in masks]
            sample["mask"] = torch.stack(masks).type(image.dtype)

        return sample
            
    def __len__(self):
        return len(self.files)