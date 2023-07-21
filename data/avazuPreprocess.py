import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


feat_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
              'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15',
              'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


class AvazuPreprocess(object):

    def __init__(self):
        super(AvazuPreprocess, self).__init__()
        self.data_dir = '../avazu/'
        self.processed_full_data_dir = os.path.join(self.data_dir, 'processed')
        self.feature_size_file_path = os.path.join(self.data_dir, 'feature_size.pkl')
        self.train_path = os.path.join(self.processed_full_data_dir, 'processed_train_data_save_path.pkl')
        self.test_path = os.path.join(self.processed_full_data_dir, 'processed_test_data_save_path.pkl')

    def load_raw_data(self):
        print('load raw train data start')
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        print('load raw train data end')
        return train_data

    def process_full_data(self):
        data = self.load_raw_data()
        print('fill nan start')
        data[feat_names] = data[feat_names].fillna('-1', )
        print('fill nan end')
        print('transformation start')
        feature_size = {}
        for feat in tqdm(feat_names):
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
            feature_size[feat] = data[feat].nunique()
        print('transformation end')
        print('data split start')
        train_data, test_data = train_test_split(data, test_size=0.1, random_state=2020)
        print('data split end')
        print('save data start')
        process_train_data_file_path = os.path.join(self.processed_full_data_dir, 'train_processed.csv')
        process_test_data_file_path = os.path.join(self.processed_full_data_dir, 'test_processed.csv')
        train_data.to_csv(process_train_data_file_path, index=False)
        test_data.to_csv(process_test_data_file_path, index=False)
        with open(self.train_path, 'wb') as f:
            pkl.dump(process_train_data_file_path, f)
        with open(self.test_path, 'wb') as f:
            pkl.dump(process_test_data_file_path, f)
        with open(self.feature_size_file_path, 'wb') as f:
            pkl.dump(feature_size, f)
        print('save data end')


if __name__ == '__main__':
    dataProcessor = AvazuPreprocess()
    dataProcessor.process_full_data()
