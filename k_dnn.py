import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

#from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical

def load_data():
    with h5py.File('detection_features.h5', 'r') as f:
        features_vgg = np.array(f['vgg'])
        features_resnet = np.array(f['resnet'])
        features_densenet = np.array(f['densenet'])
        features_inception = np.array(f['inception'])
        labels = np.array(f['labels'])
    
    features_resnet = features_resnet.reshape(features_resnet.shape[:2])
    features_inception = features_inception.reshape(features_inception.shape[:2])
    #features = np.concatenate([features_resnet, features_inception], axis = -1) 
    #print("features_resnet.shape: ", features_resnet.shape) 
    #print("features_inception.shape: ", features_inception.shape)
    #print("features_densenet.shape: ", features_densenet.shape)
    features = np.concatenate([features_resnet, features_densenet, features_inception, features_vgg], axis = -1)
    #print("features.shape: ", features.shape)
    print("labels:\n", labels)
    print("load data")

    return features_resnet, labels

def dnn_model(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size):
    inputs = Input(shape = (train_X.shape[1],))
    x = Dense(256, activation = 'relu')(inputs)
    x = Dropout(0.5)(x)
    predictions = Dense(120, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = predictions)
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    model.fit(train_X, train_Y, epochs = epoch, batch_size = batch_size, validation_data = (test_X, test_Y))
    #score = model.evaluate(test_X, test_Y, batch_size = 1)
    #pre_y = model.predict(test_X, batch_size = 1)
    #pre = np.argmax(pre_y, axis = 1)
    #true_y = np.argmax(test_Y, axis = 1)

if __name__ == "__main__":
    features, labels = load_data()
    y_ = to_categorical(labels, num_classes = 120)
    print("y_.shape: ", y_.shape)
    X_train, X_val, y_train, y_val = train_test_split(features, y_, test_size = 0.2)
    print("y_train.shape: ", y_train.shape)

    lr = 0.0001
    epochs = 50
    batch_size = 128

    dnn_model(X_train, y_train, X_val, y_val, lr, epochs, batch_size)     
    

