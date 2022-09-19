import torch
import glob
import numpy as np
from PIL import Image

dir = 'C:/Users/lyaha/OneDrive/Рабочий стол/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Loads to the GPU
x_size = (572, 572)
y_size = (572, 572)


class Data(torch.utils.data.Dataset):

    def __init__(self):
        self.x_train = sorted(glob.glob(dir + 'train/*_HC.png'))
        self.y_train = sorted(glob.glob(dir + 'train/*_HC_Mask.png'))

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.x_train[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
        y = np.array(Image.open(self.y_train[idx]).convert("L").resize(y_size)).reshape(1, 572, 572)
        return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)


