import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ID_HELD_OUT = 0.2

"""
Reference: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_anchor_year(anchor_year_group):
    year_min = int(anchor_year_group[:4])
    year_max = int(anchor_year_group[-4:])
    assert year_max - year_min == 2
    return year_min


def assign_readmission_label(row):
    curr_subject_id = row.subject_id
    curr_admittime = row.admittime

    next_row_subject_id = row.next_row_subject_id
    next_row_admittime = row.next_row_admittime

    if curr_subject_id != next_row_subject_id:
        label = 0
    elif (next_row_admittime - curr_admittime).days > 15:
        label = 0
    else:
        label = 1

    return label


def diag_icd9_to_3digit(icd9):
    if icd9.startswith('E'):
        if len(icd9) >= 4:
            return icd9[:4]
        else:
            print(icd9)
            return icd9
    else:
        if len(icd9) >= 3:
            return icd9[:3]
        else:
            print(icd9)
            return icd9


def diag_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def diag_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + diag_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + diag_icd10_to_3digit(icd[6:])
    else:
        raise


def list_join(lst):
    return ' <sep> '.join(lst)


def proc_icd9_to_3digit(icd9):
    if len(icd9) >= 3:
        return icd9[:3]
    else:
        print(icd9)
        return icd9


def proc_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def proc_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + proc_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + proc_icd10_to_3digit(icd[6:])
    else:
        raise


def process_mimic_data(data_dir):
    set_seed(seed=42)

    for file in ['patients.csv', 'diagnoses_icd.csv', 'procedures_icd.csv']:
        if not os.path.isfile(os.path.join(data_dir, file)):
            raise ValueError(f'Please download {file} to {data_dir}')

    # Patients
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    patients['real_anchor_year'] = patients.anchor_year_group.apply(lambda x: get_anchor_year(x))
    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'real_anchor_year']]
    patients = patients.dropna().reset_index(drop=True)
    admissions = pd.read_csv(os.path.join(data_dir, 'admissions.csv'))
    admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
    admissions = admissions[['subject_id', 'hadm_id', 'ethnicity', 'admittime', 'hospital_expire_flag']]
    admissions = admissions.dropna()
    admissions['mortality'] = admissions.hospital_expire_flag
    admissions = admissions.sort_values(by=['subject_id', 'hadm_id', 'admittime'])
    admissions['next_row_subject_id'] = admissions.subject_id.shift(-1)
    admissions['next_row_admittime'] = admissions.admittime.shift(-1)
    admissions['readmission'] = admissions.apply(lambda x: assign_readmission_label(x), axis=1)
    admissions = admissions[['subject_id', 'hadm_id', 'ethnicity', 'admittime', 'mortality', 'readmission']]
    admissions = admissions.dropna().reset_index(drop=True)

    # Diagnoses ICD
    diagnoses_icd = pd.read_csv(os.path.join(data_dir, 'diagnoses_icd.csv'))
    diagnoses_icd = diagnoses_icd.dropna()
    diagnoses_icd = diagnoses_icd.drop_duplicates()
    diagnoses_icd = diagnoses_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    diagnoses_icd['icd_code'] = diagnoses_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    diagnoses_icd['icd_3digit'] = diagnoses_icd.icd_code.apply(lambda x: diag_icd_to_3digit(x))
    diagnoses_icd = diagnoses_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    diagnoses_icd = diagnoses_icd.rename(columns={'icd_3digit': 'diagnoses'})

    # Procedures ICD
    procedures_icd = pd.read_csv(os.path.join(data_dir, 'procedures_icd.csv'))
    procedures_icd = procedures_icd.dropna()
    procedures_icd = procedures_icd.drop_duplicates()
    procedures_icd = procedures_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    procedures_icd['icd_code'] = procedures_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    procedures_icd['icd_3digit'] = procedures_icd.icd_code.apply(lambda x: proc_icd_to_3digit(x))
    procedures_icd = procedures_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    procedures_icd = procedures_icd.rename(columns={'icd_3digit': 'procedure'})

    # Merge
    df = admissions.merge(patients, on='subject_id', how='inner')
    df['real_admit_year'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.real_anchor_year, axis=1)
    df['age'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.anchor_age, axis=1)
    df = df[['subject_id', 'hadm_id',
             'admittime', 'real_admit_year',
             'age', 'gender', 'ethnicity',
             'mortality', 'readmission']]
    df = df.merge(diagnoses_icd, on=['subject_id', 'hadm_id'], how='inner')
    df = df.merge(procedures_icd, on=['subject_id', 'hadm_id'], how='inner')
    df.to_csv(os.path.join(data_dir, 'data_preprocessed.csv'))

    # Cohort Selection
    processed_file = os.path.join(data_dir, 'processed_mimic_data.csv')
    df = df[df.age.apply(lambda x: (x >= 18) & (x <= 89))]
    df.to_csv(processed_file, index=False)
    return processed_file


class MIMICStay:

    def __init__(self,
                 icu_id,
                 icu_timestamp,
                 mortality,
                 readmission,
                 age,
                 gender,
                 ethnicity):
        self.icu_id = icu_id    # str
        self.icu_timestamp = icu_timestamp  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity

        self.diagnosis = []     # list of tuples (timestamp in min (int), diagnosis (str))
        self.treatment = []     # list of tuples (timestamp in min (int), treatment (str))

    def __repr__(self):
        return f'MIMIC ID-{self.icu_id}, mortality-{self.mortality}, readmission-{self.readmission}'


def get_stay_dict(save_dir):
    mimic_dict = {}
    input_path = process_mimic_data(save_dir)
    fboj = open(input_path)
    name_list = fboj.readline().strip().split(',')
    for eachline in fboj:
        t = eachline.strip().split(',')
        tempdata = {eachname: t[idx] for idx, eachname in enumerate(name_list)}
        mimic_value = MIMICStay(icu_id=tempdata['hadm_id'],
                                 icu_timestamp=tempdata['real_admit_year'],
                                 mortality=tempdata['mortality'],
                                 readmission=tempdata['readmission'],
                                 age=tempdata['age'],
                                 gender=tempdata['gender'],
                                 ethnicity=tempdata['ethnicity'])
        mimic_value.diagnosis = tempdata['diagnoses'].split(' <sep> ')
        mimic_value.treatment = tempdata['procedure'].split(' <sep> ')
        mimic_dict[tempdata['hadm_id']] = mimic_value

    pickle.dump(mimic_dict, open(os.path.join(save_dir, 'mimic_stay_dict.pkl'), 'wb'))

def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_codes = dataset[year][0]['code']
        train_labels = dataset[year][0]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_codes = np.array(train_codes)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][0]['code'] = np.stack(new_train_codes, axis=0)
        dataset[year][0]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))


def preprocess_MIMIC(data, args):
    np.random.seed(0)
    ENV = [i for i in list(range(2008, 2020, 3))]
    datasets = {}
    temp_datasets = {}

    for i in ENV:
        datasets[i] = {}
        temp_datasets[i] = {'code':[], 'labels':[]}

    for eachadmit in data:
        year = int(data[eachadmit].icu_timestamp)
        if (year - 2008) % 3 > 0:
            year = 3 * int((year - 2008)/3) + 2008
        if year in temp_datasets:
            if args.prediction_type not in temp_datasets[year]:
                temp_datasets[year][args.prediction_type]=[]
            if args.prediction_type == 'mortality':
                temp_datasets[year]['labels'].append(data[eachadmit].mortality)
            elif args.prediction_type == 'readmission':
                temp_datasets[year]['labels'].append(data[eachadmit].readmission)
            dx_list = ['dx' for _ in data[eachadmit].diagnosis]
            tr_list = ['tr' for _ in data[eachadmit].treatment]
            temp_datasets[year]['code'].append([data[eachadmit].diagnosis + data[eachadmit].treatment, dx_list + tr_list])

    for eachyear in temp_datasets.keys():
        temp_datasets[eachyear]['labels'] = np.array(temp_datasets[eachyear]['labels'])
        temp_datasets[eachyear]['code'] = np.array(temp_datasets[eachyear]['code'])
        num_samples = temp_datasets[eachyear]['labels'].shape[0]
        seed_ = np.random.get_state()
        np.random.seed(0)

        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[eachyear][0] = {}
        datasets[eachyear][0]['code'] = temp_datasets[eachyear]['code'][idxs[:num_train_samples]]
        datasets[eachyear][0]['labels'] = temp_datasets[eachyear]['labels'][idxs[:num_train_samples]]

        datasets[eachyear][1] = {}
        datasets[eachyear][1]['code'] = temp_datasets[eachyear]['code'][idxs[num_train_samples:]]
        datasets[eachyear][1]['labels'] = temp_datasets[eachyear]['labels'][idxs[num_train_samples:]]

        datasets[eachyear][2] = {}
        datasets[eachyear][2]['code'] = temp_datasets[eachyear]['code']
        datasets[eachyear][2]['labels'] = temp_datasets[eachyear]['labels']

    with open(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl'),'wb') as f:
        pickle.dump(datasets, f)


def preprocess_orig(args):
    if not os.path.exists(os.path.join(args.data_dir, 'mimic_stay_dict.pkl')):
        get_stay_dict(args.data_dir)
    data = pickle.load(open(os.path.join(args.data_dir, 'mimic_stay_dict.pkl'), 'rb'))
    preprocess_MIMIC(data, args)


def preprocess(args):
    if not os.path.isfile(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)



class MIMICBase(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        preprocess(args)
        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}_wildtime.pkl'
        else:
            self.data_file = f'{str(self)}_wildtime_{args.reduced_train_prop}.pkl'
        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.num_classes = 2
        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: self.datasets[i][self.mode]['labels'].shape[0] for i in self.ENV}

        ## create a datasets object
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.input_dim = []
        cumulative_batch_size = 0
        self.task_idxs = {}
        start_idx = 0

        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'].astype(np.int) == classid)[0]
                self.class_id_list[classid][i] = sel_idx
            print(f'Year {str(i)} loaded')

            cumulative_batch_size += min(self.mini_batch_size, self.num_examples[i])
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(self.mini_batch_size, self.num_examples[i]), 3, 32, 32))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['code'] = np.concatenate(
            (self.datasets[time][self.mode]['code'], self.datasets[prev_time][self.mode]['code']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'].astype(np.int) == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['code'] = np.concatenate(
                (self.datasets[time][self.mode]['code'], self.datasets[prev_time][self.mode]['code'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'].astype(np.int) == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        codes = [item[0] for item in self.datasets[time_idx][self.mode]['code'][sel_idx]]
        types = [item[1] for item in self.datasets[time_idx][self.mode]['code'][sel_idx]]
        code = (codes, types)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx].astype(np.int)

        return code, torch.LongTensor([label]).squeeze(0).unsqueeze(1).cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return f'mimic_{self.args.prediction_type}'


class MIMIC(MIMICBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        code = self.datasets[self.current_time][self.mode]['code'][index]
        label = int(self.datasets[self.current_time][self.mode]['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class MIMICGroup(MIMICBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx = self.task_idxs[sel_time][0]
            end_idx = self.task_idxs[sel_time][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            code = self.datasets[self.current_time][self.mode]['code'][sel_idx]
            label = int(self.datasets[self.current_time][self.mode]['labels'][sel_idx])
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return code, label_tensor, group_tensor

        else:
            code = self.datasets[self.current_time][self.mode]['code'][index]
            label = int(self.datasets[self.current_time][self.mode]['labels'][index])
            label_tensor = torch.LongTensor([label])

            return code, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
