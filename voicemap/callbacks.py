import numpy as np
from tqdm import tqdm
from keras.callbacks import Callback
import voicemap.metrics as mtx


class ModelValidator(Callback):
    def __init__(self, batch_gen, metrics, monitor, mode):
        super(ModelValidator, self).__init__()
        self.batch_gen = batch_gen
        self.metrics = metrics
        self.monitor = monitor
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best_acc = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_acc = 0.
        else:
            raise AttributeError('[ERROR] ModelValidator mode %s is unknown')

    def on_train_begin(self, logs=None):
        super(ModelValidator, self).on_train_begin(logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' BEFORE TRAINING: Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        super(ModelValidator, self).on_epoch_end(epoch, logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' EPOCH %d. Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            epoch, vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        super(ModelValidator, self).on_train_end(logs)
        print('=' * 20 + ' Training report ' + '=' * 20)
        print('Best validation %s: epoch %s / %.4f\n' % (self.monitor.upper(), self.best_epoch, self.best_acc))

    @staticmethod
    def validate_model(model, batch_gen, metrics):
        """
        # Arguments
            model: Keras model
            data: BaseDataLoader
            metrics: list of metrics
        # Output
            dictionary with values of metrics and loss
        """
        cut_model = model
        # if DCASEModelTrainer.is_mfom_objective(model):
        #     input = model.get_layer(name='input').output
        #     preact = model.get_layer(name='output').output
        #     cut_model = Model(input=input, output=preact)

        n_class = cut_model.output_shape[1]
        y_true, y_pred = np.empty((0, n_class)), np.empty((0, n_class))
        loss, cnt = 0, 0

        for X_b, Y_b in batch_gen.batch():
            ps = cut_model.predict_on_batch(X_b)
            # average decision across files windows predictions
            # mean_ps = np.mean(ps, axis=0)
            # mean_y = np.mean(Y_b, axis=0)
            # TODO fix it later
            y_pred = np.vstack([y_pred, ps])
            y_true = np.vstack([y_true, Y_b])
            # NOTE: it is fake loss, caz Y is fed
            # if DCASEModelTrainer.is_mfom_objective(model):
            #     X_b = [Y_b, X_b]
            l = model.test_on_batch(X_b, Y_b)
            loss += l
            cnt += 1

        vals = {'val_loss': loss / cnt}
        print(y_pred.shape)
        print(y_true.shape)

        for m in metrics:
            if m == 'micro_f1':
                p = mtx.step(y_pred, threshold=0.5)
                vals[m] = mtx.micro_f1(y_true, p)
            elif m == 'pooled_eer':
                p = y_pred.flatten()
                y = y_true.flatten()
                vals[m] = mtx.eer(y, p)
            elif m == 'class_wise_eer':
                vals[m] = np.mean(mtx.class_wise_eer(y_true, y_pred))
            elif m == 'accuracy':
                p = np.argmax(y_pred, axis=-1)
                y = np.argmax(y_true, axis=-1)
                vals[m] = mtx.pooled_accuracy(y, p)
            else:
                raise KeyError('[ERROR] Such a metric is not implemented: %s...' % m)
        return vals


class SiameseValidator(ModelValidator):
    def __init__(self, batch_gen, num_tasks, n_shot, k_way, metrics, monitor, mode, preprocessor=lambda x: x):
        super(SiameseValidator, self).__init__(batch_gen, metrics, monitor, mode)
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.preprocessor = preprocessor

    def on_train_begin(self, logs=None):
        super(ModelValidator, self).on_train_begin(logs)
        vs = self.n_shot_task_validation(self.model, self.batch_gen, self.metrics, self.preprocessor,
                                         self.num_tasks, self.n_shot, self.k_way)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' BEFORE TRAINING: Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vs = self.n_shot_task_validation(self.model, self.batch_gen, self.metrics, self.preprocessor,
                                         self.num_tasks, self.n_shot, self.k_way)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' EPOCH %d. Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            epoch, vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = epoch

    @staticmethod
    def n_shot_task_validation(model, batch_gen, metrics, preprocessor, num_tasks, n_shot, k_way, distance='euclidean'):
        """Evaluate a siamese network on k-way, n-shot classification tasks generated from a particular dataset.

        # Arguments
            model: Model to evaluate
            dataset: Dataset (currently LibriSpeechDataset only) from which to build evaluation tasks
            preprocessor: Preprocessing function to apply to samples
            num_tasks: Number of tasks to evaluate with
            n_shot: Number of samples per class present in the support set
            k_way: Number of classes present in the support set
            network_type: Either 'siamese' or 'classifier'. This controls how to get the embedding function from the model
            distance: Either 'euclidean' or 'cosine'. This controls how to combine the support set samples for n > 1 shot
            tasks
        """
        # TODO for siamese n_shot = 1, k_way = 2
        y_true, y_pred = np.empty((0, k_way)), np.empty((0, k_way))
        loss = 0
        cnt = 0

        if n_shot == 1:
            # Directly use siamese network to get pairwise verficiation score, minimum is closest
            for i_eval in tqdm(range(num_tasks)):
                query_sample, support_set_samples = batch_gen.build_n_shot_task(k_way, n_shot)

                # k of the same utterances
                input_1 = np.stack([query_sample[0]] * k_way)[:, :, np.newaxis]
                # k different utterances, 0th is the same as in input_1
                input_2 = support_set_samples[0][:, :, np.newaxis]
                # Pass an empty list to the labels parameter as preprocessor functions on batches not samples
                ([input_1, input_2], _) = preprocessor(([input_1, input_2], []))
                pred = model.predict([input_1, input_2])
                # accumulate scores
                y_pred = np.vstack([y_pred, pred.T])
                y = np.array([0] + [1] * (k_way-1))
                y_true = np.vstack([y_true, y])  # 0: same, 1: different speaker
                # loss
                l = model.test_on_batch([input_1, input_2], y)
                loss += np.mean(l)
                cnt += 1

        vals = {'val_loss': loss / cnt}

        for m in metrics:
            if m == 'micro_f1':
                p = mtx.step(y_pred, threshold=0.5)
                vals[m] = mtx.micro_f1(y_true, p)
            elif m == 'pooled_eer':
                p = y_pred.flatten()
                y = y_true.flatten()
                vals[m] = mtx.eer(y, p)
            elif m == 'class_wise_eer':
                vals[m] = np.mean(mtx.class_wise_eer(y_true, y_pred))
            elif m == 'accuracy':
                p = np.argmax(y_pred, axis=-1)
                y = np.argmax(y_true, axis=-1)
                vals[m] = mtx.pooled_accuracy(y, p)
            else:
                raise KeyError('[ERROR] Such a metric is not implemented: %s...' % m)

        return vals
