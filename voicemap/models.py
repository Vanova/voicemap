from keras.models import Model, Sequential
from keras import layers
import keras.backend as K


def get_baseline_convolutional_encoder(embedding_dimension, input_shape, config):
    """
    input shape: [batch_sz; frame_wnd; band; channel]
    """
    print('DNN input shape', input_shape)

    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
        freq_axis = 2
    else:
        raise NotImplementedError('[ERROR] Only for TensorFlow background.')

    nb_filters = config['feature_maps']
    dropout_rate = config['dropout']
    pool_sz = [4, 2, 2]  # max-pooling across frequency only

    # Input block
    # feat_input = layers.Input(shape=input_shape, name='input')

    encoder = Sequential()
    encoder.add(layers.BatchNormalization(axis=freq_axis, name='bn_0_freq'))
    # CNN block
    for i, sz in enumerate(pool_sz):
        encoder.add(layers.Conv2D(filters=(i + 1) * nb_filters, kernel_size=(3, 3), padding='same'))
        encoder.add(layers.BatchNormalization(axis=channel_axis))
        encoder.add(layers.Activation(config['activation']))
        encoder.add(layers.MaxPool2D(pool_size=(sz, sz)))
        encoder.add(layers.Dropout(dropout_rate))
    encoder.add(layers.MaxPool2D())
    encoder.add(layers.GlobalMaxPool2D())
    encoder.add(layers.Dense(embedding_dimension))

    # Unwrap network
    # filters = 128
    # feat_input = layers.Input(shape=(48000, 1), name='input')
    # x = layers.Conv1D(filters, 3, padding='same', activation='relu')(feat_input)
    # x = layers.BatchNormalization()(x)
    # x = layers.SpatialDropout1D(dropout_rate)(x)
    # x = layers.MaxPool1D(4, 4)(x)
    #
    # # Further convs
    # x =layers.Conv1D(2 * filters, 3, padding='same', activation='relu')(x)
    # x =layers.BatchNormalization()(x)
    # x =layers.SpatialDropout1D(dropout_rate)(x)
    # x =layers.MaxPool1D()(x)
    #
    # x =layers.Conv1D(3 * filters, 3, padding='same', activation='relu')(x)
    # x =layers.BatchNormalization()(x)
    # x =layers.SpatialDropout1D(dropout_rate)(x)
    # x =layers.MaxPool1D()(x)
    #
    # x =layers.Conv1D(4 * filters, 3, padding='same', activation='relu')(x)
    # x =layers.BatchNormalization()(x)
    # x =layers.SpatialDropout1D(dropout_rate)(x)
    # x =layers.MaxPool1D()(x)
    #
    # x =layers.GlobalMaxPool1D()(x)
    #
    # encoder =layers.Dense(embedding_dimension)(x)
    return encoder


def build_siamese_net(encoder, input_shape,  distance_metric='uniform_euclidean'):
    assert distance_metric in ('uniform_euclidean', 'weighted_euclidean',
                               'uniform_l1', 'weighted_l1',
                               'dot_product', 'cosine_distance')

    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    if distance_metric == 'weighted_l1':
        # This is the distance metric used in the original one-shot paper
        # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
        embedded_distance = layers.Subtract()([encoded_1, encoded_2])
        embedded_distance = layers.Lambda(lambda x: K.abs(x))(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'uniform_euclidean':
        # Simpler, no bells-and-whistles euclidean distance
        # Still apply a sigmoid activation on the euclidean distance however
        embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
        # Sqrt of sum of squares
        embedded_distance = layers.Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)), name='euclidean_distance'
        )(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'cosine_distance':
        raise NotImplementedError
        # cosine_proximity = layers.Dot(axes=-1, normalize=True)([encoded_1, encoded_2])
        # ones = layers.Input(tensor=K.ones_like(cosine_proximity))
        # cosine_distance = layers.Subtract()([ones, cosine_proximity])
        # output = layers.Dense(1, activation='sigmoid')(cosine_distance)
    else:
        raise NotImplementedError

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese


def build_wave_siamese():
    pass