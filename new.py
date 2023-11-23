import cv2
import torch

import numpy as np
path = "1.jpg"

img = cv2.imread(path)

print(img.shape)
out1 = np.transpose(img, (2,0,1))
out2 = torch.unsqueeze(out1, 0)

print(out2.shape)