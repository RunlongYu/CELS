import time
import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]
feat_names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
              'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
              'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']


class CriteoProcessor(object):

    def __init__(self):
        super(CriteoProcessor, self).__init__()
        self.data_dir = '../criteo/'
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        self.bucket_file_dir = os.path.join(self.data_dir, 'bucket')
        self.feature_map_file_dir = os.path.join(self.data_dir, 'feature_map')
        self.feature_map_file_path = os.path.join(self.feature_map_file_dir,'feature_map.pkl')
        self.sparse_feature_bucket_file_path = os.path.join(self.bucket_file_dir,'sparse_feature_bucket.pkl')
        self.dense_feature_bucket_file_path = os.path.join(self.bucket_file_dir,'dense_feature_bucket.pkl')
        self.feature_size_file_path = os.path.join(self.data_dir,'feature_size.pkl')
        self.processed_full_train_data_file_path = os.path.join(self.processed_data_dir,
                                                                'processed_train_data_save_path.pkl')
        self.processed_full_test_data_file_path = os.path.join(self.processed_data_dir,
                                                               'processed_test_data_save_path.pkl')
    def load_raw_data(self):
        print('load raw data start')
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'train.txt'), sep='\t', names=feat_names)
        print('load raw data end')
        return raw_data

    def split_raw_data(self):
        print('split raw data start')
        train_data, test_data = train_test_split(self.raw_data, test_size=0.1, random_state=2020)
        print('split raw data end')
        return train_data, test_data

    def make_feat_map(self):
        print('make feature map start')
        if not os.path.exists(self.feature_map_file_path):
            dense_features = ['I' + str(i) for i in range(1, 14)]
            sparse_features = ['C' + str(i) for i in range(1, 27)]
            self.raw_data[dense_features] = self.raw_data[dense_features].fillna(0, )
            self.raw_data[sparse_features] = self.raw_data[sparse_features].fillna('-1', )
            feat_map = []
            for feature_names in tqdm(dense_features + sparse_features):
                dict = self.raw_data[feature_names].value_counts().to_dict()
                feat_map.append(dict)
            pkl.dump(feat_map, open(self.feature_map_file_path, 'wb'))
        print('make feature map end')

    def make_bucket(self):
        print('make bucket map start')
        if not (os.path.exists(self.sparse_feature_bucket_file_path)
                and os.path.exists(self.dense_feature_bucket_file_path)
                and os.path.exists(self.feature_size_file_path)):
            feat_map = pkl.load(open(self.feature_map_file_path, 'rb'))
            feat_sizes = {}
            num_feat = []
            for i in tqdm(range(13)):
                kv = []
                for k, v in feat_map[i].items():
                    if k == '':
                        kv.append([-1, v])
                    else:
                        kv.append([k, v])
                kv = sorted(kv, key=lambda x: x[0])
                kv = np.array(kv)
                _s = 0
                thresholds = []
                for j in range(len(kv) - 1):
                    _k, _v = kv[j]
                    _s += _v
                    if _s > 40:
                        thresholds.append(_k)
                        _s = 0
                thresholds = np.array(thresholds)
                num_feat.append(thresholds)
                feat_sizes[dense_features[i]] = len(num_feat[i]) + 1

            cat_feat = []
            for i in tqdm(range(13, 39)):
                cat_feat.append({})
                for k, v in feat_map[i].items():
                    if v > 40:
                        cat_feat[i - 13][k] = len(cat_feat[i - 13])
                cat_feat[i - 13]['other'] = len(cat_feat[i - 13])
                feat_sizes[sparse_features[i - 13]] = len(cat_feat[i - 13])
            pkl.dump(num_feat, open(self.dense_feature_bucket_file_path, 'wb'))
            pkl.dump(cat_feat, open(self.sparse_feature_bucket_file_path, 'wb'))
            pkl.dump(feat_sizes, open(self.feature_size_file_path, 'wb'))
        print('make bucket map end')

    def put_data_into_bucket(self, raw_data, save_path):
        num_feat = pkl.load(open(self.dense_feature_bucket_file_path, 'rb'))
        cat_feat = pkl.load(open(self.sparse_feature_bucket_file_path, 'rb'))
        i = 0
        start_time = time.time()
        for sparse_index, sparse in tqdm(enumerate(sparse_features)):
            raw_data[sparse] = raw_data[sparse].apply(
                lambda value: cat_feat[sparse_index][value] if value in cat_feat[sparse_index] else
                cat_feat[sparse_index][
                    'other'])
        for dense_index, dense in tqdm(enumerate(dense_features)):
            raw_data[dense] = raw_data[dense].apply(
                lambda value: len(np.where((num_feat[dense_index] < value))[0]))
        raw_data.to_csv(save_path, index=False)
        print(time.time() - start_time)

    def process_full_data(self):
        self.raw_data = self.load_raw_data()
        self.train_data, self.test_data = self.split_raw_data()
        self.make_feat_map()
        self.make_bucket()
        print('process full train data start')
        if not os.path.exists(self.processed_full_train_data_file_path):
            processed_full_train_data_filepath = os.path.join(self.processed_data_dir, 'train_processed.csv')
            self.put_data_into_bucket(self.train_data,
                                      processed_full_train_data_filepath)
            pkl.dump(processed_full_train_data_filepath,
                     open(self.processed_full_train_data_file_path, 'wb'))
        print('process full train data end')
        print('process full test data start')
        if not os.path.exists(self.processed_full_test_data_file_path):
            processed_full_test_data_filepath = os.path.join(self.processed_data_dir, 'test_processed.csv')
            self.put_data_into_bucket(self.test_data,
                                      processed_full_test_data_filepath)
            pkl.dump(processed_full_test_data_filepath,
                     open(self.processed_full_test_data_file_path, 'wb'))
        print('process full test data end')


if __name__ == '__main__':
    dataProcessor = CriteoProcessor()
    dataProcessor.process_full_data()
