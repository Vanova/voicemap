import os
import kaldi_python_io as kio
import voicemap.utils as vu


def converter_ark2hdf(data_dir, out_file):
    depends = [os.path.join(data_dir, x) for x in ['feats.scp', 'spk2utt', 'utt2spk']]
    for depend in depends:
        if not os.path.exists(depend):
            raise RuntimeError('Missing file {}!'.format(depend))

    feat_reader = kio.ScriptReader(depends[0])
    writer = vu.HDFWriter(file_name=out_file)
    cnt = 0
    for fn in feat_reader.index_keys:
        feat = feat_reader[fn]
        # dump features
        writer.append(file_id=fn, feat=feat)
        cnt += 1
        print("%d. processed: %s" % (cnt, fn))
    writer.close()


TEST_DATA_DIR = 'data/'
out_file = os.path.join(TEST_DATA_DIR, 'toy_dataset.hdf')
data_dir = '/home/vano/wrkdir/projects_data/sre_2019/toy_dataset'
converter_ark2hdf(data_dir, out_file)
