import cv2 as cv
import model
import metric
import numpy as np
from Dataset import Dataset
import tensorflow as tf

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
final_model = model.age_estimator(initializer=tf.keras.initializers.HeNormal())
final_model.load_weights('/home/sarp/Documents/CS559/Homework/FacialAgeEstimation/best_age_model.h5')
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse',
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

print("Evaluate on validation data")
test_results = final_model.evaluate(valid_images, valid_labels, batch_size=128)
print("Valid. MSE Loss: ", test_results[0], " Valid. MAE: ", test_results[1])

print("Evaluate on test data")
test_results = final_model.evaluate(test_images, test_labels, batch_size=128)
print("Test MSE Loss: ", test_results[0], " Test MAE: ", test_results[1])

print("Evaluate on test data, rounded output MAE")
test_predict = final_model.predict(test_images, batch_size=128)
reshaped_labels = test_labels.reshape((test_labels.shape[0], 1))
print("Test rounded output MAE: ", metric.rounded_mae(reshaped_labels, test_predict))
dif = np.abs(test_predict-reshaped_labels)
reshaped_dif = test_labels.reshape((dif.shape[0], ))
sorted_dif = np.argsort(reshaped_dif)
print(test_predict[sorted_dif[0]])
print(test_predict[sorted_dif[250]])
print(test_predict[sorted_dif[-1:]])

print(test_labels[sorted_dif[0]])
print(test_labels[sorted_dif[250]])
print(test_labels[sorted_dif[-1:]])

cv.imshow('best', test_images[sorted_dif[0]])
cv.waitKey(0)

cv.imshow('mid', test_images[sorted_dif[250]])
cv.waitKey(0)

cv.imshow('worst', test_images[sorted_dif[-1]])
cv.waitKey(0)



