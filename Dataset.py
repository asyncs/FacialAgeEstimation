import os
import cv2 as cv
import numpy as np
import tensorflow as tf


class Dataset:

    def __init__(self, path):
        self.path = path
        self.data = None
        self.label = None

    def get_data(self):
        path = self.path
        data_array = []
        label_array = []

        for image in os.listdir(path):
            image_name = os.fsdecode(image)
            image = cv.imread(path+image_name)[:, :, 0].reshape(91, 91, 1)
            label = int(image_name[:3])
            data_array.append(image)
            label_array.append(label)

        self.data = np.array(data_array)
        self.label = np.array(label_array)

    def form(self):
        self.get_data()
        dataset = tf.data.Dataset.from_tensor_slices((self.data, self.label))
        print(dataset)