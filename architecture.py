from keras.layers import Dense, Input, Convolution2D, Flatten, Reshape, Dropout, MaxPooling2D
from keras.applications import vgg16
from keras.models import Model, model_from_json
import os.path


def initVgg16():
    vgg = vgg16.VGG16(weights="imagenet")
    inp = Input(shape=(224, 224, 3), name='in')
    shared_layers = vgg.layers[1](inp)
    for i in range(len(vgg.layers)):
        if 1 < i < len(vgg.layers) - 5:
            shared_layers = vgg.layers[i](shared_layers)
    return inp, shared_layers


def getModel(filename):
    if os.path.isfile(filename + '.json'):
        return loadModel(filename)

    inp, shared_layers = initVgg16()
    score_predictions = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(shared_layers)
    score_predictions = Flatten()(score_predictions)
    score_predictions = Dense(512, activation='relu')(score_predictions)
    score_predictions = Dropout(0.5)(score_predictions)
    # to change in order to the number of classes
    score_predictions = Dense(10, activation='relu')(score_predictions)
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(1, name='score_out')(score_predictions)

    seg_predictions = Convolution2D(512, (1, 1), activation='relu')(shared_layers)
    seg_predictions = Flatten()(seg_predictions)
    seg_predictions = Dense(512)(seg_predictions)
    seg_predictions = Dense(56 * 56)(seg_predictions)
    seg_predictions = Reshape(target_shape=(56, 56), name='seg_out')(seg_predictions)

    model = Model(inputs=inp, outputs=[seg_predictions, score_predictions])
    return model


def loadModel(filename):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + '.h5')
    return loaded_model


def saveModel(model, filename):
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(filename + '.h5')
