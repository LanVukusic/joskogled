from tensorflow import keras
from tensorflow.keras import layers


def block_cnn(filters, kernel_size, strides, max_pooling):
    l = []
    l.append(layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same'
    ))
    l.append(layers.ReLU())
    l.append(layers.BatchNormalization())
    if max_pooling:
        l.append(layers.MaxPool2D())
    return l


def model1(image_shape):
    input_l_cc = keras.Input(shape=image_shape)
    input_l_mlo = keras.Input(shape=image_shape)
    input_r_cc = keras.Input(shape=image_shape)
    input_r_mlo = keras.Input(shape=image_shape)

    cnn = [
        [
            *block_cnn( 16, 3, 2, False),
            *block_cnn( 16, 3, 1, True ),
            *block_cnn( 32, 3, 1, False)
        ],
        [
            *block_cnn( 32, 3, 1, True ),
            *block_cnn( 64, 3, 1, False)
        ]
    ]
    global_pool = layers.GlobalAveragePooling2D()

    # calculate through cnn and save mid points
    x0_out = [input_l_cc]
    x1_out = [input_l_mlo]
    x2_out = [input_r_cc]
    x3_out = [input_r_mlo]
    for part in cnn:
        x0, x1, x2, x3 = x0_out[-1], x1_out[-1], x2_out[-1], x3_out[-1]
        for layer in part:
            x0 = layer(x0)
            x1 = layer(x1)
            x2 = layer(x2)
            x3 = layer(x3)
        x0_out.append(x0)
        x1_out.append(x1)
        x2_out.append(x2)
        x3_out.append(x3)
    
    # remove first item - input
    x0_out = x0_out[1:]
    x1_out = x1_out[1:]
    x2_out = x2_out[1:]
    x3_out = x3_out[1:]
    
    # add global pooling to every out
    for i in range(len(cnn)):
        x0_out[i] = global_pool(x0_out[i])
        x1_out[i] = global_pool(x1_out[i])
        x2_out[i] = global_pool(x2_out[i])
        x3_out[i] = global_pool(x3_out[i])

    x = layers.Concatenate(axis=-1)([*x0_out, *x1_out, *x2_out, *x3_out])

    
    """cnn = [
        *block_cnn( 16, 3, 2, False),
        *block_cnn( 16, 3, 1, True ),
        *block_cnn( 32, 3, 2, False),
        *block_cnn( 32, 3, 1, True ),
        *block_cnn( 64, 3, 1, False),
        #*block_cnn( 64, 3, 1, True ),
        #*block_cnn(128, 3, 1, False),
        #*block_cnn(128, 3, 1, True ),
        layers.GlobalAveragePooling2D()
    ]

    x0, x1, x2, x3 = input_l_cc, input_l_mlo, input_r_cc, input_r_mlo
    for l in cnn:
        x0 = l(x0)
        x1 = l(x1)
        x2 = l(x2)
        x3 = l(x3)
    

    x = layers.Concatenate(axis=-1)([x0, x1, x2, x3])
    """#x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(units=1, activation='sigmoid')(x)

    outputs = x

    return keras.Model(
        inputs=(input_l_cc, input_l_mlo, input_r_cc, input_r_mlo),
        outputs=outputs
    )


def model_kaggle(image_shape):
    inputs = keras.Input(shape=image_shape)

    cnn = [
        [
            *block_cnn( 16, 3, 2, False),
            *block_cnn( 16, 3, 1, True ),
            *block_cnn( 32, 3, 1, False)
        ],
        [
            *block_cnn( 32, 3, 1, True ),
            *block_cnn( 64, 3, 1, False)
        ]
    ]
    global_pool = layers.GlobalAveragePooling2D()

    # calculate through cnn and save mid points
    x0_out = [inputs]
    for part in cnn:
        x0 = x0_out[-1]
        for layer in part:
            x0 = layer(x0)
        x0_out.append(x0)
    
    # remove first item - input
    x0_out = x0_out[1:]
    
    # add global pooling to every out
    for i in range(len(cnn)):
        x0_out[i] = global_pool(x0_out[i])

    x = layers.Concatenate(axis=-1)([*x0_out])

    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(units=1, activation='sigmoid')(x)

    outputs = x

    return keras.Model(
        inputs=inputs,
        outputs=outputs
    )
