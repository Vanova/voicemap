import os
import h5py
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from keras.utils import Sequence
from kaldi_python_io import Reader, ScriptReader
import voicemap.utils.io as vio
import config as cfg


class HDFDataGenerator(Sequence):
    """This class subclasses the Keras Sequence object. The __getitem__ function will return a raw audio sample and it's
    label.

    This class also contains functionality to build verification tasks and n-shot, k-way classification tasks.

    # Arguments
        subsets: What LibriSpeech datasets to include.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        label: One of {speaker, sex}. Whether to use sex or speaker ID as a label.
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
        cache: bool. Whether or not to use the cached index file
    """

    def __init__(self, data_dir, wnd_size, stochastic=True, multiple_cache=False):
        self.data_dir = data_dir
        self.stochastic = stochastic
        print('Initialising SREDatasetGenerator with {} frames.'.format(wnd_size))

        depends = [os.path.join(data_dir, x) for x in ['feats.scp', 'spk2utt', 'utt2spk']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        self.wnd_size = wnd_size
        self.spk2utt = Reader(depends[1], num_tokens=-1)
        self.utt2spk = Reader(depends[2], num_tokens=-1)

        hdf_file = os.path.join(data_dir, 'feats.hdf')
        if not os.path.exists(hdf_file):
            print('[INFO] Need to cache features to hdf format...')
            self.ark2hdf_caching(scp_file=depends[0], hdf_file=hdf_file)

        self.feat_reader = h5py.File(hdf_file, 'r')

        uall = []
        sall = []
        for u, s in self.utt2spk.index_dict.items():
            uall.append(u)
            sall.append(s[0])
        self.df = pd.DataFrame(columns=['speaker_id', 'file_id'])
        self.df['speaker_id'] = sall
        self.df['file_id'] = uall
        self.nunique_spks = len(self.df['speaker_id'].unique())
        print('Finished indexing data. {} usable files found.'.format(len(self)))

    def __getitem__(self, index):
        fn = self.df['file_id'][index]
        instance = np.array(self.feat_reader[fn], dtype=np.float32)
        # Choose a random sample of the file
        if self.stochastic:
            ut_len = instance.shape[0] - self.wnd_size
            if ut_len > 0:
                start = np.random.randint(0, ut_len)
                chunk = instance[start:start + self.wnd_size]
            else:
                chunk = np.pad(instance, ((-ut_len, 0), (0, 0)), 'edge')
        else:
            start = 0
            chunk = instance[start:start + self.wnd_size]
        label = self.df['speaker_id'][index]
        return chunk, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    def get_alike_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        alike_pairs = pd.merge(
            self.df.sample(num_pairs * 2),
            self.df,
            on='speaker_id'
        ).sample(num_pairs)[['speaker_id', 'file_id_x', 'file_id_y']]

        alike_pairs = zip(alike_pairs['file_id_x'].values, alike_pairs['file_id_y'].values)

        return alike_pairs

    def get_differing_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to different speakers."""
        # First get a random sample from the dataset and then get a random sample from the remaining part of the dataset
        # that doesn't contain any speakers from the first random sample
        random_sample = self.df.sample(num_pairs)
        random_sample_from_other_speakers = self.df[~self.df['speaker_id'].isin(
            random_sample['speaker_id'])].sample(num_pairs)

        differing_pairs = zip(random_sample['file_id'].values, random_sample_from_other_speakers['file_id'].values)

        return differing_pairs

    def build_verification_batch(self, batchsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        # Take only the instances not labels and stack to form a batch of pairs of instances from the same speaker
        alike_pairs = self.get_alike_pairs(batchsize // 2)
        l = list(zip(*alike_pairs))
        idx1 = [self.df[self.df['file_id'] == id].index[0] for id in l[0]]
        idx2 = [self.df[self.df['file_id'] == id].index[0] for id in l[1]]

        input_1_alike = np.stack([self[i][0] for i in idx1])
        input_2_alike = np.stack([self[i][0] for i in idx2])

        # Take only the instances not labels and stack to form a batch of pairs of instances from different speakers
        differing_pairs = self.get_differing_pairs(batchsize // 2)
        l = list(zip(*differing_pairs))
        idx1 = [self.df[self.df['file_id'] == id].index[0] for id in l[0]]
        idx2 = [self.df[self.df['file_id'] == id].index[0] for id in l[1]]
        input_1_different = np.stack([self[i][0] for i in idx1])
        input_2_different = np.stack([self[i][0] for i in idx2])

        # Merge utterances
        input_1 = np.vstack([input_1_alike, input_1_different])[:, :, :, np.newaxis]
        input_2 = np.vstack([input_2_alike, input_2_different])[:, :, :, np.newaxis]

        outputs = np.append(np.zeros(batchsize // 2), np.ones(batchsize // 2))[:, np.newaxis]

        return [input_1, input_2], outputs

    def yield_verification_batches(self, batchsize):
        """Convenience function to yield verification batches forever."""
        while True:
            ([input_1, input_2], labels) = self.build_verification_batch(batchsize)
            yield ([input_1, input_2], labels)

    def build_n_shot_task(self, k, n=1):
        """
        This method builds a k-way n-shot classification task. It returns a support set of n audio samples each from k
        unique speakers. In addition it will return a query sample. Downstream models will attempt to match the query
        sample to the correct samples in the support set.
        :param k: Number of unique speakers to include in this task
        :param n: Number of audio samples to include from each speaker
        :return:
        """
        if k >= self.nunique_spks:
            raise (ValueError, 'k must be smaller than the number of unique speakers in this dataset!')

        if k <= 1:
            raise (ValueError, 'k must be greater than or equal to one!')

        query = self.df.sample(1)
        query_sample = self[query.index.values[0]]

        is_query_speaker = self.df['speaker_id'] == query['speaker_id'].values[0]
        not_same_sample = self.df.index != query.index.values[0]
        correct_samples = self.df[is_query_speaker & not_same_sample].sample(n)

        # Sample k-1 speakers
        # TODO: weight by length here
        other_support_set_speakers = np.random.choice(
            self.df[~is_query_speaker]['speaker_id'].unique(), k - 1, replace=False)

        other_support_samples = []
        for i in range(k - 1):
            is_same_speaker = self.df['speaker_id'] == other_support_set_speakers[i]
            other_support_samples.append(
                self.df[~is_query_speaker & is_same_speaker].sample(n)
            )
        support_set = pd.concat([correct_samples] + other_support_samples)
        support_set_samples = tuple(np.stack(i) for i in zip(*[self[i] for i in support_set.index]))

        return query_sample, support_set_samples

    @staticmethod
    def ark2hdf_caching(scp_file, hdf_file):
        ark_reader = ScriptReader(scp_file)
        writer = vio.HDFWriter(file_name=hdf_file)
        cnt = 0
        for fn in ark_reader.index_keys:
            feat = ark_reader[fn]
            # dump features
            writer.append(file_id=fn, feat=feat)
            cnt += 1
            print("%d. processed: %s" % (cnt, fn))
        writer.close()


class WavDataGenerator(Sequence):
    """This class subclasses the Keras Sequence object.
        The __getitem__ function will return a raw audio sample and it's label.

        This class also contains functionality to
        build verification tasks and n-shot, k-way classification tasks.

    # Arguments
        subsets: What LibriSpeech datasets to include.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        label: One of {speaker, sex}. Whether to use sex or speaker ID as a label.
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
        cache: bool. Whether or not to use the cached index file
    """

    def __init__(self, data_dir, subsets, seconds, stochastic=True, pad=False, cache=True):
        self.data_dir = data_dir
        self.subset = subsets
        self.fragment_seconds = seconds
        self.wnd_length = int(seconds * cfg.SRE_SAMPLING_RATE)
        self.stochastic = stochastic
        self.pad = pad

        print('Initialising WaveSREDataGenerator with minimum length = {}s and subsets = {}'.format(seconds, subsets))

        if isinstance(subsets, str):
            subsets = [subsets]

        cached_df = []
        found_cache = {s: False for s in subsets}
        if cache:
            for s in subsets:
                subset_index_path = os.path.join(cfg.PATH, 'data/%s.index.csv' % s)
                if os.path.exists(subset_index_path):
                    cached_df.append(pd.read_csv(subset_index_path))
                    found_cache[s] = True

        # Index the remaining subsets if any
        if all(found_cache.values()) and cache:
            self.meta_data = pd.concat(cached_df)
        else:

            df = pd.DataFrame()
            for subset, found in found_cache.items():
                if not found:
                    subset_path = os.path.join(self.data_dir, subset)
                    tmp_df = self.index_subset(subset_path)
                    tmp_df['subset'] = [subset] * len(tmp_df)
                    # Merge individual audio files with indexing dataframe
                    df = df.append(tmp_df)

            # Concatenate with already existing dataframe if any exist
            self.meta_data = pd.concat(cached_df + [df])

        # Save index files to data folder
        for s in subsets:
            self.meta_data[self.meta_data['subset'] == s].to_csv(cfg.PATH + '/data/{}.index.csv'.format(s), index=False)

        # Trim too-small files
        if not self.pad:
            self.meta_data = self.meta_data[self.meta_data['length'] > self.fragment_seconds]

        # Index of dataframe has direct correspondence to item in dataset
        self.meta_data = self.meta_data.reset_index(drop=True)
        self.unique_speakers = len(self.meta_data['speaker_id'].unique())
        print('Finished indexing data. %d usable utterances found. %d unique speakers.' % (
        len(self), self.unique_speakers))

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        full_fpath = os.path.join(self.data_dir, 'toy_dataset', 'wav', self.meta_data['file_path'][index])
        # full_fpath = self.meta_data['file_path'][index]
        pipe = eval(self.meta_data['pipeline'][index])
        ch = int(pipe[5])
        assert ch in (1, 2)

        instance, samplerate = sf.read(full_fpath)
        instance = instance if len(instance.shape) == 1 else instance[:, ch - 1]
        # Choose a random sample of the file
        if self.stochastic:
            start = np.random.randint(0, max(len(instance) - self.wnd_length, 1))
        else:
            start = 0

        instance = instance[start:start + self.wnd_length]

        # Check for required length and pad if necessary
        if self.pad and len(instance) < self.wnd_length:
            less_timesteps = self.wnd_length - len(instance)
            if self.stochastic:
                # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                # before and the appropriate number of 0s after the instance
                less_timesteps = self.wnd_length - len(instance)

                before_len = np.random.randint(0, less_timesteps)
                after_len = less_timesteps - before_len

                instance = np.pad(instance, (before_len, after_len), 'constant')
            else:
                # Deterministic padding. Append 0s to reach self.fragment_length
                instance = np.pad(instance, (0, less_timesteps), 'constant')

        label = self.meta_data['speaker_id'][index]

        return instance, label

    def get_alike_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        alike_pairs = pd.merge(
            self.meta_data.sample(num_pairs * 2, weights='length'),
            self.meta_data,
            on='speaker_id'
        ).sample(num_pairs)[['speaker_id', 'file_id_x', 'file_id_y']]
        alike_pairs = zip(alike_pairs['file_id_x'].values, alike_pairs['file_id_y'].values)
        return alike_pairs

    def get_differing_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to different speakers."""
        # First get a random sample from the dataset and then get a random sample from the remaining part of the dataset
        # that doesn't contain any speakers from the first random sample
        rnd_smp = self.meta_data.sample(num_pairs, weights='length')
        rnd_smp_other_spkrs = self.meta_data[~self.meta_data['speaker_id'].isin(
            rnd_smp['speaker_id'])].sample(num_pairs, weights='length')

        differing_pairs = zip(rnd_smp['id'].values, rnd_smp_other_spkrs['id'].values)
        return differing_pairs

    def build_verification_batch(self, batchsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        # Take only the instances not labels and stack to form a batch of pairs of instances from the same speaker
        alike_pairs = self.get_alike_pairs(batchsize // 2)
        l = list(zip(*alike_pairs))
        idx1 = [self.meta_data[self.meta_data['file_id'] == id].index[0] for id in l[0]]
        idx2 = [self.meta_data[self.meta_data['file_id'] == id].index[0] for id in l[1]]

        input_1_alike = np.stack([self[i][0] for i in idx1])
        input_2_alike = np.stack([self[i][0] for i in idx2])

        # Take only the instances not labels and stack to form a batch of pairs of instances from different speakers
        differing_pairs = self.get_differing_pairs(batchsize // 2)
        l = list(zip(*differing_pairs))
        idx1 = [self.meta_data[self.meta_data['file_id'] == id].index[0] for id in l[0]]
        idx2 = [self.meta_data[self.meta_data['file_id'] == id].index[0] for id in l[1]]
        input_1_different = np.stack([self[i][0] for i in idx1])
        input_2_different = np.stack([self[i][0] for i in idx2])

        # Merge utterances
        input_1 = np.vstack([input_1_alike, input_1_different])[:, :, np.newaxis]
        input_2 = np.vstack([input_2_alike, input_2_different])[:, :, np.newaxis]

        outputs = np.append(np.zeros(batchsize // 2), np.ones(batchsize // 2))[:, np.newaxis]
        return [input_1, input_2], outputs

    def yield_verification_batches(self, batchsize):
        """Convenience function to yield verification batches forever."""
        while True:
            ([input_1, input_2], labels) = self.build_verification_batch(batchsize)
            yield ([input_1, input_2], labels)

    def build_n_shot_task(self, k, n=1):
        """
        This method builds a k-way n-shot classification task. It returns a support set of n audio samples each from k
        unique speakers. In addition it will return a query sample. Downstream models will attempt to match the query
        sample to the correct samples in the support set.
        :param k: Number of unique speakers to include in this task
        :param n: Number of audio samples to include from each speaker
        :return:
        """
        if k >= self.unique_speakers:
            raise (ValueError, 'k must be smaller than the number of unique speakers in this dataset!')

        if k <= 1:
            raise (ValueError, 'k must be greater than or equal to one!')

        query = self.meta_data.sample(1, weights='length')
        query_sample = self[query.index.values[0]]

        is_query_speaker = self.meta_data['speaker_id'] == query['speaker_id'].values[0]
        not_same_sample = self.meta_data.index != query.index.values[0]
        correct_samples = self.meta_data[is_query_speaker & not_same_sample].sample(n, weights='length')

        # Sample k-1 speakers
        # TODO: weight by length here
        other_support_set_speakers = np.random.choice(
            self.meta_data[~is_query_speaker]['speaker_id'].unique(), k - 1, replace=False)

        other_support_samples = []
        for i in range(k - 1):
            is_same_speaker = self.meta_data['speaker_id'] == other_support_set_speakers[i]
            other_support_samples.append(
                self.meta_data[~is_query_speaker & is_same_speaker].sample(n, weights='length')
            )
        support_set = pd.concat([correct_samples] + other_support_samples)
        support_set_samples = tuple(np.stack(i) for i in zip(*[self[i] for i in support_set.index]))

        return query_sample, support_set_samples

    @staticmethod
    def index_subset(subset_path):
        """
        Index a subset by looping through all of it's files and recording their speaker ID, filepath and length.
        :param subset_path: Name of the subset
        :return: A list of dicts containing information about all the audio files in a particular subset of the dataset
        """
        print('Indexing %s...' % subset_path)

        depends = [os.path.join(subset_path, x) for x in ['utt2spk', 'wav.scp']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        utt2spk = Reader(depends[0], num_tokens=-1)
        wavscp = vio.load_dictionary(depends[1], delim=' ')

        assert set(wavscp.keys()) == set(utt2spk.index_keys)

        # format of wav.scp: {<file_id> : ['sph2pipe', '-f', 'wav', '-p', '-c', '2', '<file_path>', '|']}
        uall = []
        sall = []
        fpaths = []
        pipes = []
        for u, s in utt2spk.index_dict.items():
            if len(wavscp[u]) == 8:
                uall.append(u)
                sall.append(s[0])
                fpaths.append(wavscp[u][6])
                pipes.append(wavscp[u])
            else:
                print('[Warning] unknown pipline format for %s: %s' % (u, ' '.join(wavscp[u])))

        df = pd.DataFrame(columns=['file_id', 'speaker_id', 'file_path', 'pipeline', 'samples', 'length'])
        df['file_id'] = uall
        df['speaker_id'] = sall
        df['file_path'] = fpaths
        df['pipeline'] = pipes

        # walk through dataset and calculate samples and length
        progress_bar = tqdm(total=df['file_id'].size)
        for fp in df['file_path']:
            full_fpath = os.path.join(subset_path, 'wav', fp)
            # full_fpath = fp
            if not os.path.isfile(full_fpath):
                print('[Warning] file does not exist: %d' % full_fpath)
                continue

            signal, fs = sf.read(full_fpath)
            assert fs == cfg.SRE_SAMPLING_RATE

            df['samples'][df['file_path'] == fp] = len(signal)
            df['length'][df['file_path'] == fp] = len(signal) / float(cfg.SRE_SAMPLING_RATE)

            progress_bar.update(1)
        progress_bar.close()
        assert not df.isnull().any().any()
        return df
