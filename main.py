import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import os
import pyttsx3

# download and load mnist model


def load_model():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    batch_size = 128
    epochs = 15
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)
    return model


# open video stream from usb camera
cap = cv.VideoCapture("/dev/video2")
# set frame rate
cap.set(cv.CAP_PROP_FPS, 30)
# text to speech
engine = pyttsx3.init()
# cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Unable to open video stream')
    exit(0)

# load model from file or train new model
if os.path.exists('modelmnist.h5'):
    model = keras.models.load_model('modelmnist.h5')
else:
    model = load_model()
    model.save('modelmnist.h5')
while True:
    # read frame
    ret, frame = cap.read()
    if not ret:
        print('Unable to read frame')
        exit(0)

    # gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # threshold
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    # find contours
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # draw contours
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # predict
    predominant = -1
    aux = 0
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 120 and h > 120:
            roi = thresh[y:y+h, x:x+w]
            roi = cv.resize(roi, (28, 28))
            roi = roi.reshape(1, 28, 28, 1)
            # draw rectangle
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # scale 0/1
            roi = roi / 255.0
            # classify
            pred = model.predict(roi)
            # get predominant class
            aux = pred
            if type(predominant) == int:
                predominant = pred
            if aux[0][aux.argmax()] > predominant[0][predominant.argmax()]:
                predominant = pred

    if type(predominant) == int:
        cv.putText(frame, "Error", (20, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv.putText(frame, "La probabilidad  de ser "+str(predominant.argmax())+" es de " +
                   str(predominant[0][predominant.argmax()]), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        engine.say(str(predominant.argmax()))
        #engine.setProperty('rate', 1000)
        # engine.runAndWait()

    # display frame
    cv.imshow('frameBW', thresh)
    if cv.waitKey(1) == ord('q'):
        break

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    # close video stream
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
