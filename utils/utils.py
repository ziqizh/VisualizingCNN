import os
import requests
import json

import torch

"""
this file is utils lib
"""
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        print("image size", img.shape)
        # img[:,:,0] = (img[:,:,0] * self.std[0]) + self.mean[0]
        # img[:,:,1] = img[:,:,1] * self.std[1] + self.mean[1]
        # img[:,:,2] = img[:,:,2] * self.std[2] + self.mean[2]
        img[:,:,0] = (img[:,:,0] * (img[:,:,0].max() - img[:,:,0].min())) + img[:,:,0].min()
        img[:,:,1] = (img[:,:,1] * (img[:,:,1].max() - img[:,:,1].min())) + img[:,:,1].min()
        img[:,:,2] = (img[:,:,2] * (img[:,:,2].max() - img[:,:,2].min())) + img[:,:,2].min()
        return img

def decode_predictions(preds, top=5):
    """Decode the prediction of an ImageNet model

    # Arguments
        preds: torch tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return

    # Return
        A list of lists of top class prediction tuples
        One list of turples per sample in batch input.

    """


    class_index_path = 'https://s3.amazonaws.com\
/deep-learning-models/image-models/imagenet_class_index.json'

    class_index_dict = None

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predciton'
        '(i.e. a 2D array of shape (samples, 1000)).'
        'Found array with shape: ' + str(preds.shape)
        )

    if not os.path.exists('./data/imagenet_class_index.json'):
        r = requests.get(class_index_path).content
        with open('./data/imagenet_class_index.json', 'w+') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)

    results = []
    for pred in preds:
        top_value, top_indices = torch.topk(pred, top)
        result = [tuple(class_index_dict[str(i.item())]) + (pred[i].item(),) \
                for i in top_indices]
        result = [tuple(class_index_dict[str(i.item())]) + (j.item(),) \
        for (i, j) in zip(top_indices, top_value)]
        results.append(result)

    return results
