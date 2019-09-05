import os
import multiprocessing
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from voicemap.callbacks import SiameseValidator
from voicemap.utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
from voicemap.models import get_baseline_convolutional_encoder, build_siamese_net
from voicemap.sre_2016 import SREDataGenerator
from config import LIBRISPEECH_SAMPLING_RATE, PATH

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############
# Parameters #
##############
n_seconds = 3
window_size = 170
downsampling = 1  # 4
batchsize = 32  # 64
filters = 128
embedding_dimension = 64
dropout = 0.1
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
pad = True
num_epochs = 200
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

val_metrics = ['pooled_eer', 'accuracy', 'micro_f1', 'class_wise_eer']

# Derived parameters
fbanks = 64
input_dimension = (window_size, fbanks, 1)
param_str = 'siamese__filters_{}__embed_{}__drop_{}__pad={}'.format(filters, embedding_dimension, dropout, pad)

###################
# Create datasets #
###################
# TODO replace with Kaldi
data_dir = '/home/vano/wrkdir/projects_data/sre_2019/toy_dataset'
train = SREDataGenerator(data_dir, window_size, stochastic=True)
valid = SREDataGenerator(data_dir, window_size, stochastic=False)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
train_generator = (batch for batch in train.yield_verification_batches(batchsize))
valid_generator = (batch for batch in valid.yield_verification_batches(batchsize))

################
# Define model #
################
net_config = {
      'activation': 'elu',
      'dropout': 0.1,
      'feature_maps': filters,
}
encoder = get_baseline_convolutional_encoder(embedding_dimension, input_dimension, config=net_config)
siamese = build_siamese_net(encoder, input_dimension, distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
siamese.summary()

#################
# Training Loop #
#################
callbacks = [
    # First generate custom n-shot classification metric
    NShotEvaluationCallback(
        num_evaluation_tasks, n_shot_classification, k_way_classification, valid),
    SiameseValidator(batch_gen=valid,
                     num_tasks=num_evaluation_tasks,
                     n_shot=1,
                     k_way=2,  # number of speakers sampled
                     metrics=val_metrics,
                     monitor='pooled_eer',
                     mode='min'),
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
    EarlyStopping(
        monitor='pooled_eer',
        patience=15,
        verbose=1,
        mode='min',
        min_delta=0.001),
    TensorBoard(
        log_dir=os.path.join(PATH, 'logs'),
        write_graph=True)
]

siamese.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train) // batchsize,
    validation_data=valid_generator,
    validation_steps=100,
    epochs=num_epochs,
    workers=multiprocessing.cpu_count(),
    verbose=1,
    use_multiprocessing=True,
    callbacks=callbacks
)
