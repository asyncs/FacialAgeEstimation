import itertools
import tensorflow as tf
import pickle


def conv_block(filter_no, filter_size, pooling):
    layers = []

    conv_layer = tf.keras.layers.Conv2D(filters=filter_no, kernel_size=filter_size, padding='same',
                                        kernel_regularizer=tf.keras.regularizers.L2(5e-4))
    layers.append(conv_layer)
    layers.append(tf.keras.layers.ReLU())
    #layers.append(tf.keras.layers.LeakyReLU(alpha=0.1))
    layers.append(tf.keras.layers.BatchNormalization())
    if pooling:
        layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    return layers


def conv_block_trail(layer_no, filter_no, filter_size, batch_norm, pooling):
    layers = []
    for layer in range(layer_no):
        conv_layer = tf.keras.layers.Conv2D(filters=filter_no[layer], kernel_size=filter_size[layer], padding='same')
        layers.append(conv_layer)
        if batch_norm:
            batch_layer = tf.keras.layers.BatchNormalization()
            layers.append(batch_layer)
            relu_layer = tf.keras.layers.ReLU()
            layers.append(relu_layer)
            if pooling and (layer + 1) % 2 == 0:
                max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
                layers.append(max_pool_layer)
        else:
            if pooling and (layer + 1) % 2 == 0:
                relu_layer = tf.keras.layers.ReLU()
                layers.append(relu_layer)
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
    conv_layer = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), padding='same')
    layers.append(conv_layer)
    flatten_layer = tf.keras.layers.Flatten()
    layers.append(flatten_layer)

    fc1_layer = tf.keras.layers.Dense(64)
    layers.append(fc1_layer)
    relu_layer = tf.keras.layers.ReLU()
    layers.append(relu_layer)
    output = tf.keras.layers.Dense(1)
    layers.append(output)

    return layers


def model(pooling, layer_no, batch_norm, filter_size, filter_no):
    custom_model = tf.keras.Sequential()
    custom_model.add(tf.keras.Input(shape=(91, 91, 1)))

    conv_block_layers = conv_block_trail(layer_no=layer_no, filter_no=filter_no, filter_size=filter_size,
                                         batch_norm=batch_norm, pooling=pooling)
    for layer in conv_block_layers:
        custom_model.add(layer)

    fc_layers = regression_block()
    for layer in fc_layers:
        custom_model.add(layer)

    return custom_model


def get_all_models():
    models_params = []
    layer_no = [2, 3, 4]
    max_pooling = [True, False]
    batch_norm = [True, False]
    filter_sizes = [(3, 3), (4, 4)]
    number_of_filters = [32, 64]

    pooling_1 = [p[0] for p in itertools.product(max_pooling, repeat=1)]
    norm_1 = [p[0] for p in itertools.product(batch_norm, repeat=1)]
    fsize_1 = [p[0] for p in itertools.product(filter_sizes, repeat=1)]
    fno_1 = [p[0] for p in itertools.product(number_of_filters, repeat=1)]

    for any_pooling in pooling_1:
        for any_norm in norm_1:
            for any_fsize in fsize_1:
                for any_fno in fno_1:
                    models_params.append([1, [any_fno], [any_fsize], any_norm, any_pooling])
    for number_of_layers in layer_no:

        pooling = [p[0] for p in itertools.product(max_pooling, repeat=1)]
        norm = [p[0] for p in itertools.product(batch_norm, repeat=1)]
        fsize = [p for p in itertools.product(filter_sizes, repeat=number_of_layers)]
        fno = [p for p in itertools.product(number_of_filters, repeat=number_of_layers)]

        for any_pooling in pooling:
            for any_norm in norm:
                for any_fsize in fsize:
                    any_fsize = list(any_fsize)
                    for any_fno in fno:
                        any_fno = list(any_fno)
                        models_params.append([number_of_layers, any_fno, any_fsize, any_norm, any_pooling])
    return models_params


def explore(train_images, train_labels, valid_images, valid_labels, model_params):
    errors = []
    for i in range(model_params.shape[0]):
        param = model_params[i]
        layer_no = param[0]
        filter_no = param[1]
        filter_size = param[2]
        batch_norm = param[3]
        pooling = param[4]
        test_model = model(pooling=pooling, layer_no=layer_no, batch_norm=batch_norm, filter_size=filter_size,
                           filter_no=filter_no)
        test_model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = test_model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=25,
                                 batch_size=64, verbose=0)

        print("Model no: ", i, "Min Error: ", min(history.history['val_mean_absolute_error']))
        errors.append(min(history.history['val_mean_absolute_error']))

    with open('valid_explore_error', 'wb') as fp:
        pickle.dump(errors, fp)


def age_estimator():
    custom_model = tf.keras.Sequential()
    custom_model.add(tf.keras.Input(shape=(91, 91, 1)))

    conv_layers = conv_block(16, (3, 3), pooling=True)
    for layer in conv_layers:
        custom_model.add(layer)

    conv_layers = conv_block(32, (3, 3), pooling=False)
    for layer in conv_layers:
        custom_model.add(layer)

    conv_layers = conv_block(64, (3, 3), pooling=True)
    for layer in conv_layers:
        custom_model.add(layer)

    conv_layers = conv_block(128, (3, 3), pooling=True)
    for layer in conv_layers:
        custom_model.add(layer)


    flatten_layer = tf.keras.layers.Flatten()
    custom_model.add(flatten_layer)

    fc_layer = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(0.001))
    custom_model.add(fc_layer)
    custom_model.add(tf.keras.layers.BatchNormalization())
    custom_model.add(tf.keras.layers.ReLU())
    custom_model.add(tf.keras.layers.Dropout(0.15))

    fc_layer = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L2(0.001))
    custom_model.add(fc_layer)
    custom_model.add(tf.keras.layers.BatchNormalization())
    custom_model.add(tf.keras.layers.ReLU())
    custom_model.add(tf.keras.layers.Dropout(0.15))

    fc_layer = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L2(0.001))
    custom_model.add(fc_layer)
    custom_model.add(tf.keras.layers.BatchNormalization())
    custom_model.add(tf.keras.layers.ReLU())
    custom_model.add(tf.keras.layers.Dropout(0.15))

    output = tf.keras.layers.Dense(1)
    custom_model.add(output)

    return custom_model
