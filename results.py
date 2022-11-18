import os
import matplotlib.pyplot as plt
import pickle
import model
import numpy as np
from Dataset import Dataset
import tensorflow as tf


def get_result(result_model, t_images, t_labels, v_images, v_labels):
    result_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse',
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = result_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                               batch_size=128)

    return history


def find_general_structure():
    model_params = model.get_all_models()
    with open('/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/explore_error', 'rb') as fp:
        explore_errors = pickle.load(fp)

    x = range(1360)

    fig = plt.figure()
    plt.scatter(x[:16], explore_errors[:16], c='b', marker='o', label='1 layer')
    plt.scatter(x[16:80], explore_errors[16:80], c='r', marker='o', label='2 layer')
    plt.scatter(x[80:336], explore_errors[80:336], c='g', marker='o', label='3 layer')
    plt.scatter(x[336:1360], explore_errors[336:1360], c='m', marker='o', label='4 layer')
    plt.legend(loc='upper left')
    plt.xlabel('Model No')
    plt.ylabel('Min Error')
    plt.show()

    min_model_index = np.argmin(np.array(explore_errors))
    print("Min. model index: ", min_model_index)
    print("Min. model params: ", model_params[min_model_index])
    print("Min. error: ", explore_errors[min_model_index])


def find_pooling_loc(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [[True, True, True, True], [False, True, True, True], [True, False, True, True],
               [True, True, False, True],
               [True, True, True, False]]
    batch_norm = [True, True, True, True]
    filters = [16, 32, 64, 128]
    count = 0
    for each in pooling:
        e_model = model.generic_model(each, batch_norm, filters)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['All pooling layers are active', '1st layer inactive', '2nd layer inactive', '3rd layer inactive',
                    '4th layer inactive']
    fig = plt.figure()

    for index in range(5):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_batchnorm_loc(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [False, True, True, True]
    batch_norm = [[True, True, True, True], [False, True, True, True], [True, False, True, True],
                  [True, True, False, True],
                  [True, True, True, False]]
    filters = [16, 32, 64, 128]
    count = 0
    for each in batch_norm:
        e_model = model.generic_model(pooling, each, filters)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['All batch-norm layers are active', '1st layer inactive', '2nd layer inactive',
                    '3rd layer inactive', '4th layer inactive']
    fig = plt.figure()

    for index in range(5):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_filter_direction(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [False, True, True, True]
    batch_norm = [True, False, True, True]
    filters = [[16, 32, 64, 128], [64, 64, 64, 64], [128, 64, 32, 16]]
    count = 0
    for each in filters:
        e_model = model.generic_model(pooling, batch_norm, each)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['Forward', 'Neutral', 'Backward']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_filter_size(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [False, True, True, True]
    batch_norm = [True, False, True, True]
    filters = [[8, 16, 32, 64], [16, 32, 64, 128], [32, 64, 128, 256]]
    count = 0
    for each in filters:
        e_model = model.generic_model(pooling, batch_norm, each)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['Small', 'Medium', 'Big']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_regression_layer(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [False, True, True, True]
    batch_norm = [True, False, True, True]
    filters = [16, 32, 64, 128]
    regression = [1, 2, 3]
    count = 0
    for each in regression:
        e_model = model.generic_model_regression(pooling, batch_norm, filters, each)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['1 FC Layer', '2 FC Layer', '3 FC Layer']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_regression_hidden(t_images, t_labels, v_images, v_labels):
    errors = []
    pooling = [False, True, True, True]
    batch_norm = [True, False, True, True]
    filters = [16, 32, 64, 128]
    regression = [[64, 32, 16], [128, 64, 32], [256, 128, 64]]
    count = 0
    for each in regression:
        e_model = model.generic_model_regression_size(pooling, batch_norm, filters, each)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['Small FC', 'Medium FC', 'Big FC']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_regression_batch(t_images, t_labels, v_images, v_labels):
    errors = []
    batch = [[True, True, True], [False, True, True], [True, False, True], [True, True, False]]

    count = 0
    for each in batch:
        e_model = model.generic_model_regression_batchnorm(batch=each)
        history = get_result(e_model, t_images, t_labels, v_images, v_labels)
        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['All batch-norm layers are active', '1st layer inactive', '2nd layer inactive',
                    '3rd layer inactive']
    fig = plt.figure()

    for index in range(4):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_loss(t_images, t_labels, v_images, v_labels):
    errors = []
    losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanSquaredError()]
    count = 0

    for each in losses:
        e_model = model.age_estimator(initializer=tf.keras.initializers.GlorotUniform())
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=each,
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['MAE Loss', 'MSE Loss']
    fig = plt.figure()

    for index in range(2):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_init(t_images, t_labels, v_images, v_labels):
    errors = []
    init = [tf.keras.initializers.RandomNormal(), tf.keras.initializers.GlorotUniform(),
            tf.keras.initializers.HeNormal()]
    count = 0

    for each in init:
        e_model = model.age_estimator(initializer=each)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['Gaussian', 'Xavier', 'He']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_cnn_weight_decay(t_images, t_labels, v_images, v_labels):
    errors = []
    cnn_weight = [1e-4, 5e-4]
    count = 0

    for each in cnn_weight:
        e_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal(), cnn_l2=each)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['1e-4', '5e-4']
    fig = plt.figure()

    for index in range(2):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_mlp_weight_decay(t_images, t_labels, v_images, v_labels):
    errors = []
    mlp_weight = [1e-3, 1e-4]
    count = 0

    for each in mlp_weight:
        e_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal(), fc_l2=each, cnn_l2=1e-4)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['1e-3', '1e-4']
    fig = plt.figure()

    for index in range(2):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_dropout(t_images, t_labels, v_images, v_labels):
    errors = []
    dropout = [0.15, 0.3, 0.5]
    count = 0

    for each in dropout:
        e_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal(), fc_dropout=each,  fc_l2=1e-4, cnn_l2=1e-4)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['0.15', '0.3', '0.5']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_batchsize(t_images, t_labels, v_images, v_labels):
    errors = []
    batchsize = [64, 128, 256]
    count = 0

    for each in batchsize:
        e_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal(), fc_l2=1e-4, cnn_l2=1e-4)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=each)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['64', '128', '256']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


def find_learningrate(t_images, t_labels, v_images, v_labels):
    errors = []
    learningrate = [1e-2, 1e-3, 1e-4]
    count = 0

    for each in learningrate:
        e_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal(), fc_l2=1e-4, cnn_l2=1e-4)
        e_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=each), loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = e_model.fit(t_images, t_labels, validation_data=(v_images, v_labels), epochs=150,
                              batch_size=128)

        count += 1
        errors.append(history.history['val_mean_absolute_error'])

    legend_names = ['1e-2', '1e-3', '1e-4']
    fig = plt.figure()

    for index in range(3):
        print("Model no: ", index, "Min ValidationError: ", min(errors[index]))
        plt.plot(errors[index], label=legend_names[index])

    plt.xlabel('Epoch No')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.show()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/training_set/"
valid_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/validation_set/"
test_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/test_set/"

train_set = Dataset(train_path)
train_images, train_labels = train_set.get_data()

valid_set = Dataset(valid_path)
valid_images, valid_labels = valid_set.get_data()

# find_general_structure()
# find_pooling_loc(train_images, train_labels, valid_images, valid_labels)
# find_batchnorm_loc(train_images, train_labels, valid_images, valid_labels)
# find_filter_direction(train_images, train_labels, valid_images, valid_labels)
# find_filter_size(train_images, train_labels, valid_images, valid_labels)
# find_regression_layer(train_images, train_labels, valid_images, valid_labels)
# find_regression_hidden(train_images, train_labels, valid_images, valid_labels)
# find_regression_batch(train_images, train_labels, valid_images, valid_labels)
# find_loss(train_images, train_labels, valid_images, valid_labels)
# find_init(train_images, train_labels, valid_images, valid_labels)
# find_cnn_weight_decay(train_images, train_labels, valid_images, valid_labels)
# find_mlp_weight_decay(train_images, train_labels, valid_images, valid_labels)
# find_dropout(train_images, train_labels, valid_images, valid_labels)
# find_batchsize(train_images, train_labels, valid_images, valid_labels)
# find_learningrate(train_images, train_labels, valid_images, valid_labels)