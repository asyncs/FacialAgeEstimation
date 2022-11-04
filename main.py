import tensorflow as tf
from Dataset import Dataset

if __name__ == '__main__':
    train_path = "UTKFace_downsampled/training_set/"

    train_set = Dataset(train_path)
    train_set.form()
