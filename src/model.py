import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from utils import MODEL_TYPE_CONTRASTIVE, MODEL_TYPE_LEGACY

INPUT_SHAPE = (100, 100, 1)
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_CONTRASTIVE_MARGIN = 1.0


def _conv_feature_stack(input_shape):
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
    bias_init = keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)
    return [
        layers.Input(shape=input_shape),
        layers.Conv2D(
            64,
            (10, 10),
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=regularizers.l2(2e-4),
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            128,
            (7, 7),
            activation="relu",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            kernel_regularizer=regularizers.l2(2e-4),
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            128,
            (4, 4),
            activation="relu",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            kernel_regularizer=regularizers.l2(2e-4),
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            256,
            (4, 4),
            activation="relu",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            kernel_regularizer=regularizers.l2(2e-4),
        ),
        layers.Flatten(),
    ]


def build_legacy_encoder(input_shape=INPUT_SHAPE):
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
    bias_init = keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)
    return keras.Sequential(
        _conv_feature_stack(input_shape)
        + [
            layers.Dense(
                4096,
                activation="sigmoid",
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=regularizers.l2(1e-3),
            ),
        ],
        name="shared_convnet",
    )


def build_siamese_network(input_shape=INPUT_SHAPE, learning_rate=6e-5, compile_model=True):
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
    bias_init = keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)

    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")
    encoder = build_legacy_encoder(input_shape=input_shape)

    encoded_l = encoder(left_input)
    encoded_r = encoder(right_input)
    abs_difference = layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]), name="abs_difference")(
        [encoded_l, encoded_r]
    )
    prediction = layers.Dense(1, activation="sigmoid", bias_initializer=bias_init, kernel_initializer=kernel_init)(
        abs_difference
    )

    model = keras.Model(inputs=[left_input, right_input], outputs=prediction, name="siamese_net")
    if compile_model:
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
    return model


def build_embedding_encoder(input_shape=INPUT_SHAPE, embedding_dim=DEFAULT_EMBEDDING_DIM):
    kernel_init = keras.initializers.HeNormal()
    encoder_input = layers.Input(shape=input_shape, name="embedding_input")
    x = encoder_input
    for layer in _conv_feature_stack(input_shape)[1:]:
        x = layer(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_initializer=kernel_init,
        kernel_regularizer=regularizers.l2(1e-4),
        name="embedding_dense",
    )(x)
    x = layers.Dense(
        embedding_dim,
        activation=None,
        kernel_initializer=kernel_init,
        kernel_regularizer=regularizers.l2(1e-4),
        name="embedding_projection",
    )(x)
    embedding = layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1), name="normalized_embedding")(x)
    return keras.Model(inputs=encoder_input, outputs=embedding, name="shared_encoder")


def _euclidean_distance(tensors):
    left, right = tensors
    squared = tf.reduce_sum(tf.square(left - right), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(squared, tf.keras.backend.epsilon()))


def contrastive_loss(margin=DEFAULT_CONTRASTIVE_MARGIN):
    margin = tf.constant(float(margin), dtype=tf.float32)

    def loss(y_true, distances):
        y_true = tf.cast(y_true, distances.dtype)
        positive_term = y_true * tf.square(distances)
        negative_term = (1.0 - y_true) * tf.square(tf.maximum(margin - distances, 0.0))
        return tf.reduce_mean(positive_term + negative_term)

    return loss


def build_contrastive_siamese_network(
    input_shape=INPUT_SHAPE,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    margin=DEFAULT_CONTRASTIVE_MARGIN,
    learning_rate=1e-4,
    compile_model=True,
):
    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")
    encoder = build_embedding_encoder(input_shape=input_shape, embedding_dim=embedding_dim)

    encoded_l = encoder(left_input)
    encoded_r = encoder(right_input)
    distance = layers.Lambda(_euclidean_distance, name="pair_distance")([encoded_l, encoded_r])

    model = keras.Model(inputs=[left_input, right_input], outputs=distance, name="contrastive_siamese_net")
    if compile_model:
        model.compile(
            loss=contrastive_loss(margin=margin),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        )
    return model


def build_model_for_type(
    model_type,
    input_shape=INPUT_SHAPE,
    compile_model=True,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    contrastive_margin=DEFAULT_CONTRASTIVE_MARGIN,
):
    if model_type == MODEL_TYPE_LEGACY:
        return build_siamese_network(input_shape=input_shape, compile_model=compile_model)
    if model_type == MODEL_TYPE_CONTRASTIVE:
        return build_contrastive_siamese_network(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            margin=contrastive_margin,
            compile_model=compile_model,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def get_encoder_for_model(model, model_type):
    if model_type == MODEL_TYPE_LEGACY:
        return model.get_layer("shared_convnet")
    if model_type == MODEL_TYPE_CONTRASTIVE:
        return model.get_layer("shared_encoder")
    raise ValueError(f"Unsupported model type: {model_type}")
