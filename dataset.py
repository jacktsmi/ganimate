import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageData(Dataset):
    def __init__(self, dirs=["dataA/dataA", "dataB/BitmojiDataset/images"], split="train", transform=None):
        self.transform = transform
        self.files_A = []
        self.files_B = []
        
        if split == "train":
          for filename in os.listdir(dirs[0]):
              base, ext = os.path.splitext(filename)
              try:
                  filenum = int(base)
              except ValueError:
                  continue
              
              if 0 <= filenum < 1899:
                  self.files_A.append(dirs[0] + '/' + filename)
          for filename in os.listdir(dirs[1]):
            base, ext = os.path.splitext(filename)
            try:
                filenum = int(base)
            except ValueError:
                continue
            
            if 0 <= filenum < 1899:
                self.files_B.append(dirs[1] + '/' + filename)
                # end if
            # end for files in root_dir
        else:
          for filename in os.listdir(dirs[0]):
              base, ext = os.path.splitext(filename)
              try:
                  filenum = int(base)
              except ValueError:
                  continue
              
              if 1899 <= filenum < 1999:
                  self.files_A.append(dirs[0] + '/' + filename)
          for filename in os.listdir(dirs[1]):
            base, ext = os.path.splitext(filename)
            try:
                filenum = int(base)
            except ValueError:
                continue
            
            if 1899 <= filenum < 1999:
                self.files_B.append(dirs[1] + '/' + filename)
                # end if
            # end for files in root_dir

    # len overload, returns the number of files in the loader
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))