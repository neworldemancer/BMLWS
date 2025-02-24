"""
Utility for the Inception mdodel
"""
import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from PIL import Image


def get_inception_files():
    idir = 'inception'
    return (os.path.join(idir, 'tensorflow_inception_graph.pb'),
            os.path.join(idir, 'imagenet_comp_graph_label_strings.txt'))


def get_inception_model():
    """Get a pretrained inception network.

    Returns
    -------
    net : tuple
        (graph_def, labels)
        where the graph_def is a tf.GraphDef and the labels
        map an integer label id (0-1000) to a list of names
    """
    # Download the trained net
    modelf, labelsf = get_inception_files()

    # Parse the ids and synsets
    txt = open(labelsf).readlines()
    labels = [(key, val.strip()) for key, val in enumerate(txt)]

    # Load the saved graph
    with gfile.GFile(modelf, 'rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('reading error')

    return graph_def, labels


def prepare_training_img(img, crop=True, resize=True, img_size=(256, 256)):
    if img.dtype != np.uint8:
        img *= 255.0

    if crop:
        crop = np.min(img.shape[:2])
        r = (img.shape[0] - crop) // 2
        c = (img.shape[1] - crop) // 2
        cropped = img[r: r + crop, c: c + crop]
    else:
        cropped = img

    if resize:
        img_pil = Image.fromarray(cropped)
        img_pil = img_pil.resize(img_size, Image.ANTIALIAS)
        rsz = np.array(img_pil.convert('RGB'))
        # rsz = imresize(cropped, img_size, preserve_range=True)
    else:
        rsz = cropped

    if rsz.ndim == 2:
        rsz = rsz[..., np.newaxis]
    if rsz.shape[2] == 4:
        rsz = rsz[..., :3]
    if rsz.shape[2] == 1:
        rsz = np.concatenate((rsz, rsz, rsz), axis=2)

    rsz = rsz.astype(np.float32)
    # subtract imagenet mean
    return (rsz - 117)


def training_img_to_display(img):
    return np.clip(img + 117, 0, 255).astype(np.uint8) 