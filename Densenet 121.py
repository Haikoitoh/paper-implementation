from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import (
    AvgPool2D,
    GlobalAveragePooling2D,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate


def conv_layer(x, filters, kernel=1, strides=1):

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel, strides=strides, padding="same")(x)
    return x


def dense_block(x, repetition, filters):

    for _ in range(repetition):
        y = conv_layer(x, 4 * filters)
        y = conv_layer(y, filters, 3)
        x = concatenate([y, x])
    return x


def transition_layer(x):

    x = conv_layer(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding="same")(x)
    return x


def densenet(input_shape, n_classes, filters=32):

    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding="same")(x)

    for repetition in [6, 12, 24, 16]:

        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)
    return model


input_shape = 224, 224, 3
n_classes = 3

model = densenet(input_shape, n_classes, filters=32)
# model = DenseNet121()
model.summary()
