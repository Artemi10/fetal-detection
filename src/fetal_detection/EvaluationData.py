import torch
import glob
import numpy as np
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU
x_size = (572, 572)
y_size = (572, 572)

class EvaluationData(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):

        self.x_train = sorted(glob.glob(path))

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.x_train[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
        return torch.from_numpy(x).float().to(device)

