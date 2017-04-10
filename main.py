from architecture import getModel, saveModel
from dataProcessing import prepareAllData
from keras.optimizers import SGD
from keras import backend as K


def binary_regression_error(y_true, y_pred):
    return 1./32 * K.log(1 + K.exp(-y_true*y_pred))


def mask_binary_regression_error(y_true, y_pred):
    return 0.5 * (1 - y_true[0][0][0]) * K.mean(K.log(1 + K.exp(-y_true*y_pred)))


model = getModel('none')
sgd = SGD(lr=0.001, decay=0.00005, momentum=0.9, nesterov=True, clipvalue=500)
model.compile(optimizer=sgd, loss={'score_out': binary_regression_error, 'seg_out': mask_binary_regression_error},metrics=['acc'])

inputs, masks, scores = prepareAllData(1000, ['outdoor', 'food', 'indoor', 'appliance', 'sports', 'person', 'animal',
                                              'vehicle', 'furniture', 'accessory'], offset=0)

model.fit({'in': inputs}, {'score_out': scores, 'seg_out': masks}, epochs=1, batch_size=32, verbose=1, shuffle=True)
saveModel(model, "deepmask10000")
