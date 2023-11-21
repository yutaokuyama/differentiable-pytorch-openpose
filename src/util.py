import numpy as np
import torch

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = torch.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = torch.concatenate((pad_up, img_padded), axis=0)
    pad_left = torch.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = torch.concatenate((pad_left, img_padded), axis=1)
    pad_down = torch.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = torch.concatenate((img_padded, pad_down), axis=0)
    pad_right = torch.tile(
        img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = torch.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def padRightDownCorner_np(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def extract_keypoint_from_candidatae_detected_result(result):
    candidate, subset = result
    keypoints = []
    if len(subset) == 0:
        return None
    for i in range(18):
        index = int(subset[0][i])
        position = [-1.0, -1.0] if index == -1 else candidate[index][0:2]
        keypoints.append(position)

    return keypoints


# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(
            weights_name.split('.')[1:])]
    return transfered_model_weights
