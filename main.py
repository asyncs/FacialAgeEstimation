import os
import pickle
import model
import visualization
import metric
import numpy as np
from Dataset import Dataset
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    train_path = "UTKFace_downsampled/training_set/"
    valid_path = "UTKFace_downsampled/validation_set/"
    test_path = "UTKFace_downsampled/test_set/"

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

    with open('explore_error', 'rb') as fp:
        explore_errors = pickle.load(fp)

    # visualization.explore_visual(model_params, explore_errors)
    train_images_norm = train_images/255.0
    valid_images_norm = valid_images/255.0

    final_model = model.age_estimator()
    final_model.summary()
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = final_model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=300,
                              batch_size=128)

    print("Evaluate on test data")
    test_results = final_model.evaluate(test_images, test_labels, batch_size=128)
    print("Test MSE Loss: ", test_results[0], " Test MAE: ", test_results[1])
    print("Min Validation Error: ", min(history.history['val_mean_absolute_error']))

    #test_prediction = final_model(test_images)
    #print(metric.rounded_mae(test_labels, test_prediction))