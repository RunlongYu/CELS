import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


feat_names = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
              'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags',
              'app_first_class', 'app_second_class', 'age', 'city', 'city_rank',
              'device_name', 'device_size', 'career', 'gender', 'net_type',
              'residence', 'his_app_size', 'his_on_shelf_time', 'app_score',
              'emui_dev', 'list_time', 'device_price', 'up_life_duration',
              'up_membership_grade', 'membership_life_duration', 'consume_purchase',
              'communication_onlinerate', 'communication_avgonline_30d', 'indu_name',
              'pt_d']


class HuaweiPreprocess(object):

    def __init__(self):
        super(HuaweiPreprocess, self).__init__()
        self.data_dir = '../huawei/'
        self.processed_full_data_dir = os.path.join(self.data_dir, 'processed')
        self.feature_size_file_path = os.path.join(self.data_dir, 'feature_size.pkl')
        self.train_path = os.path.join(self.processed_full_data_dir,
                                       'processed_train_data_save_path.pkl')
        self.test_path = os.path.join(self.processed_full_data_dir,
                                      'processed_test_data_save_path.pkl')

    def load_raw_data(self):
        print('load raw train data start')
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train_data.csv'), sep='|')
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
    dataProcessor = HuaweiPreprocess()
    dataProcessor.process_full_data()
