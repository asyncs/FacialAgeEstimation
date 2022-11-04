import tensorflow as tf


def conv_block(layer_no, filter_no, filter_size, batch_norm, pooling):
    layers = []
    for layer in range(layer_no):
        conv_layer = tf.keras.layers.Conv2D(filters=filter_no[layer], kernel_size=filter_size[layer], padding='same')
        layers.append(conv_layer)
        if batch_norm[layer]:
            batch_layer = tf.keras.layers.BatchNormalization()
            layers.append(batch_layer)
            relu_layer = tf.keras.layers.ReLU()
            layers.append(relu_layer)
        if pooling[layer]:
            max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
            layers.append(max_pool_layer)
        else:
            relu_layer = tf.keras.layers.ReLU()
            layers.append(relu_layer)

    return layers


def regression_block():
    layers = []
    conv_layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), padding='same')
    layers.append(conv_layer)
    flatten_layer = tf.keras.layers.Flatten()
    layers.append(flatten_layer)

    fc2_layer = tf.keras.layers.Dense(128)
    layers.append(fc2_layer)
    relu_layer = tf.keras.layers.ReLU()
    layers.append(relu_layer)
    output = tf.keras.layers.Dense(1)
    layers.append(output)

    return layers


def model(pooling, layer_no, batch_norm, filter_size, filter_no):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(91, 91, 1)))

    conv_block_layers = conv_block(layer_no=layer_no, filter_no=filter_no, filter_size=filter_size, batch_norm=batch_norm, pooling=pooling)
    for layer in conv_block_layers:
      model.add(layer)

    fc_layers = regression_block()
    for layer in fc_layers:
      model.add(layer)

    model.summary()

    return model