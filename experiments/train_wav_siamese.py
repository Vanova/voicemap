import os
import multiprocessing
from keras.optimizers import Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from voicemap.callbacks import SiameseValidator
from voicemap.utils.net_utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
import voicemap.wav_models as WM
from voicemap.sre_2016 import HDFDataGenerator, WavDataGenerator
import config as cfg

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############
# Parameters #
##############
n_seconds = 3
window_size = 170
downsampling = 4
batchsize = 32  # 64
filters = 128
embedding_dimension = 64
dropout = 0.1
pad = True
num_epochs = 200
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

val_metrics = ['pooled_eer', 'accuracy', 'micro_f1']

# Derived parameters
input_length = int(cfg.SRE_SAMPLING_RATE * n_seconds / downsampling)
param_str = 'siamese__filters_{}__embed_{}__drop_{}__pad={}'.format(filters, embedding_dimension, dropout, pad)

###################
# Create datasets #
###################
# === debug
train_set = 'toy_dataset'
val_set = 'toy_dataset'
data_dir = '/home/vano/wrkdir/projects_data/sre_2019/'
# === training
# train_set = 'swbd_sre_small_fbank'
# val_set = 'swbd_sre_small_fbank'
# data_dir = '/home/vano/wrkdir/projects_data/sre_2019/'

train = WavDataGenerator(data_dir, train_set, n_seconds, stochastic=True, pad=pad)
valid = WavDataGenerator(data_dir, val_set, n_seconds, stochastic=False, pad=pad)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))
valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))

################
# Define model #
################
encoder = WM.get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
siamese = WM.build_siamese_net(encoder, input_shape=(input_length, 1), distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
# opt = RMSprop()
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
siamese.summary()

#################
# Training Loop #
#################
callbacks = [
    # First generate custom n-shot classification metric
    # NShotEvaluationCallback(
    #     num_evaluation_tasks, n_shot_classification, k_way_classification, valid),
    SiameseValidator(batch_gen=valid,
                     num_tasks=num_evaluation_tasks,
                     n_shot=1,
                     k_way=2,  # number of speakers sampled
                     metrics=val_metrics,
                     monitor='pooled_eer',
                     mode='min',
                     preprocessor=batch_preprocessor),
    # Then log and checkpoint
    CSVLogger(os.path.join(cfg.PATH, 'logs/{}.csv'.format(param_str))),
    ModelCheckpoint(
        os.path.join(cfg.PATH, 'models/{}.hdf5'.format(param_str)),
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
        log_dir=os.path.join(cfg.PATH, 'logs'),
        write_graph=True)
]

siamese.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train) // batchsize,
    validation_data=valid_generator,
    validation_steps=100,
    epochs=num_epochs,
    workers=multiprocessing.cpu_count(),
    verbose=2,
    use_multiprocessing=True,
    callbacks=callbacks
)
