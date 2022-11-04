import tensorflow as tf
import itertools
import layers
from Dataset import Dataset

if __name__ == '__main__':
    train_path = "UTKFace_downsampled/training_set/"

    #train_set = Dataset(train_path)
    #train_set.form()
    number_of_layers = [1, 2, 3, 4, 5, 6]
    max_pooling = [True, False]
    batch_norm = [True, False]
    filter_sizes = [(3, 3), (4, 4)]
    number_of_filters = [32, 64]

    pooling_1 = [p[0] for p in itertools.product(max_pooling, repeat=1)]
    norm_1 = [p[0] for p in itertools.product(batch_norm, repeat=1)]
    fsize_1 = [p[0] for p in itertools.product(filter_sizes, repeat=1)]
    fno_1 = [p[0] for p in itertools.product(number_of_filters, repeat=1)]

    conv_layer_no = 1
    for any_pooling in pooling_1:
        for any_norm in norm_1:
            for any_fsize in fsize_1:
                for any_fno in fno_1:
                    pass
                    layers.model(layer_no=conv_layer_no, filter_no=[any_fno], filter_size=[any_fsize], batch_norm=[any_norm], pooling=[any_pooling])

    pooling_2 = [p for p in itertools.product(max_pooling, repeat=2)]
    norm_2 = [p for p in itertools.product(batch_norm, repeat=2)]
    fsize_2 = [p for p in itertools.product(filter_sizes, repeat=2)]
    fno_2 = [p for p in itertools.product(number_of_filters, repeat=2)]

    #pooling_12 = [[True, True], ]
