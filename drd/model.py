from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPool2D,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model


def res_block(x, filters, stage):
    x_copy = x
    f1, f2, f3 = filters

    x = Conv2D(
        f1,
        (1, 1),
        strides=(1, 1),
        name=f"res_{stage}_conv_a",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = MaxPool2D((2, 2))(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_conv_a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"res_{stage}_conv_b",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_conv_b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=f"res_{stage}_conv_c",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_conv_c")(x)

    x_copy = Conv2D(
        f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=f"res_{stage}_conv_copy",
        kernel_initializer=glorot_uniform(seed=0),
    )(x_copy)
    x_copy = MaxPool2D((2, 2))(x_copy)
    x_copy = BatchNormalization(axis=3, name=f"bn_{stage}_conv_copy")(x_copy)

    x = Add()([x, x_copy])
    x = Activation("relu")(x)

    x_copy = x
    x = Conv2D(
        f1,
        (1, 1),
        strides=(1, 1),
        name=f"res_{stage}_identity_1_a",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_1_a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"res_{stage}_identity_1_b",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_1_b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=f"res_{stage}_identity_1_c",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_1_c")(x)

    x = Add()([x, x_copy])
    x = Activation("relu")(x)

    x_copy = x
    x = Conv2D(
        f1,
        (1, 1),
        strides=(1, 1),
        name=f"res_{stage}_identity_2_a",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_2_a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"res_{stage}_identity_2_b",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_2_b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=f"res_{stage}_identity_2_c",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=f"bn_{stage}_identity_2_c")(x)

    x = Add()([x, x_copy])
    x = Activation("relu")(x)
    return x


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> Model:
    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        name="conv1",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = res_block(x, filters=[64, 64, 256], stage=2)
    x = res_block(x, filters=[128, 128, 512], stage=3)
    x = res_block(x, filters=[256, 256, 1024], stage=4)

    x = AveragePooling2D((2, 2), name="average_pooling")(x)
    x = Flatten()(x)
    x = Dense(
        num_classes,
        activation="softmax",
        name="dense_final",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)

    model = Model(inputs=x_input, outputs=x, name="ResNet18")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

