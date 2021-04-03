"""
Wrapper for FastCNN Re-ID model

Usage example:

img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
extr = Extractor("checkpoint/ckpt.t7")
feature = extr(img)
print(feature.shape)

"""

import logging

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from .model import Net

log = logging.getLogger(__name__)


class Extractor:
    """ Wrapper for FastCNN Re-ID model """

    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            "net_dict"
        ]
        self.net.load_state_dict(state_dict)
        log.info("Loading weights from %s... Done!", model_path)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, im_crops):
        """ To ensure unified input for inference """

        def _resize(img, size):
            return cv2.resize(img.astype(np.float32) / 255.0, size)

        im_batch = torch.cat(
            [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0
        ).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
