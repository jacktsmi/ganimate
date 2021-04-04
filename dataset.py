import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageData(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.files = []

        # 80-20 train-test split
        if split == 'train':
            for filename in os.listdir(root_dir):
                base, ext = os.path.splitext(filename)
                try:
                    filenum = int(base)
                except ValueError:
                    continue
                
                if 0 <= filenum < 1600:
                    self.files.append(root_dir + '/' + filename)
                # end if
            # end for files in root_dir
        # end if train split

        else:
            for filename in os.listdir(root_dir):
                base, ext = os.path.splitext(filename)
                try:
                    filenum = int(base)
                except ValueError:
                    continue
                
                if 1600 <= filenum <= 1999:
                    self.files.append(root_dir + '/' + filename)
                # end if
            # end for files in root_dir
        # end test split
    # end __init__()

    # len overload, returns the number of files in the loader
    def __len__(self):
        return len(self.files)
    # end __len__()

    # bracket [] overload
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = np.asarray(img)
        if self.transform != None:
            img = self.transform(img)
        return img
    # end __getitem__()