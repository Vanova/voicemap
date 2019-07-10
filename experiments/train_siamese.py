import os
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from voicemap.callbacks import SiameseValidator
import multiprocessing

from voicemap.utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
from voicemap.models import get_baseline_convolutional_encoder, build_siamese_net
from voicemap.librispeech import LibriSpeechDataset
from config import LIBRISPEECH_SAMPLING_RATE, PATH

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
filters = 128
embedding_dimension = 64
dropout = 0.0
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
pad = True
num_epochs = 50
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

val_metrics = ['pooled_eer', 'accuracy', 'micro_f1']

# Derived parameters
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
param_str = 'siamese__filters_{}__embed_{}__drop_{}__pad={}'.format(filters, embedding_dimension, dropout, pad)

###################
# Create datasets #
###################
data_dir = '/home/vano/wrkdir/datasets/LibriSpeech'
train = LibriSpeechDataset(data_dir, training_set, n_seconds, pad=pad)
valid = LibriSpeechDataset(data_dir, validation_set, n_seconds, stochastic=False, pad=pad)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))
valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))

################
# Define model #
################
encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# plot_model(siamese, show_shapes=True, to_file=PATH + '/plots/siamese.png')
print siamese.summary()

#################
# Training Loop #
#################
callbacks = [
    # First generate custom n-shot classification metric
    NShotEvaluationCallback(
        num_evaluation_tasks, n_shot_classification, k_way_classification, valid,
        preprocessor=batch_preprocessor),
    SiameseValidator(batch_gen=valid,
                     num_tasks=num_evaluation_tasks,
                     n_shot=1,
                     k_way=2, # number of speakers sampled
                     metrics=val_metrics,
                     monitor='pooled_eer',
                     mode='min',
                     preprocessor=batch_preprocessor),
    # Then log and checkpoint
    CSVLogger(os.path.join(PATH, 'logs/{}.csv'.format(param_str))),
    ModelCheckpoint(
        os.path.join(PATH, 'models/{}.hdf5'.format(param_str)),
        monitor='pooled_eer',
        mode='min',
        save_best_only=True,
        verbose=True),
    ReduceLROnPlateau(
        monitor='pooled_eer',
        mode='min',
        verbose=1),
    TensorBoard(
        log_dir=os.path.join(PATH, 'logs'),
        write_graph=True)
]

siamese.fit_generator(
    generator=train_generator,
    steps_per_epoch=evaluate_every_n_batches,
    validation_data=valid_generator,
    validation_steps=100,
    epochs=num_epochs,
    workers=multiprocessing.cpu_count(),
    verbose=1,
    use_multiprocessing=True,
    callbacks=callbacks
)
