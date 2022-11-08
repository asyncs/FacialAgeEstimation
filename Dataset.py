import os
import cv2 as cv
import numpy as np


class Dataset:

    def __init__(self, path):
        self.path = path

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

        return np.array(data_array), np.array(label_array)
