import sys
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from Training import pixel_accuracy, mIoU

img = torch.load(sys.argv[1])
mask = to_tensor(np.load(sys.argv[2]))
acc = pixel_accuracy(img,mask)
iou = mIoU(img,mask,5)
print('Accuracy: {}, IoU: {}'.format(acc,iou))
