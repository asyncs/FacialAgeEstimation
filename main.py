import os
import pickle
import model
import metric
import numpy as np
from Dataset import Dataset
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    train_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/training_set/"
    valid_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/validation_set/"
    test_path = "/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/UTKFace_downsampled/test_set/"

    train_set = Dataset(train_path)
    train_images, train_labels = train_set.get_data()
    noise = np.random.normal(0, 1, train_labels.shape[0])
    noise_labels = train_labels + noise

    valid_set = Dataset(valid_path)
    valid_images, valid_labels = valid_set.get_data()

    test_set = Dataset(test_path)
    test_images, test_labels = test_set.get_data()

    model_params = model.get_all_models()

    # model.explore(train_images, train_labels, valid_images, valid_labels, model_params)

    final_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal())
    es = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=200)
    mc = tf.keras.callbacks.ModelCheckpoint('best_age_model.h5', monitor='val_mean_absolute_error', mode='min', verbose=1, save_best_only=True)
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = final_model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=2000,
                              batch_size=128, callbacks=[es, mc])

    print("Evaluate on test data")
    test_results = final_model.evaluate(test_images, test_labels, batch_size=128)
    print("Test MSE Loss: ", test_results[0], " Test MAE: ", test_results[1])
    print("Min Validation Error: ", min(history.history['val_mean_absolute_error']))

    #test_prediction = final_model(test_images)
    #print(metric.rounded_mae(test_labels, test_prediction))