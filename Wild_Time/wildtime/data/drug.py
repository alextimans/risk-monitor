import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tdc import BenchmarkGroup
from torch.utils import data

from .utils import download_detection

ID_HELD_OUT = 0.1

class TdcDtiDgBase(data.Dataset):
    def __init__(self, args):
        super().__init__()

        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}_preprocessed.pkl'
        else:
            self.data_file = f'{str(self)}_preprocessed_{args.reduced_train_prop}.pkl'

        download_detection(args.data_dir, self.data_file)
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.args = args
        self.ENV = [i for i in list(range(2013, 2021))]
        self.num_tasks = 8
        self.input_shape = [(26, 100), (63, 1000)]
        self.num_classes = 1
        self.current_time = 0
        self.mode = 0

        self.task_idxs = {}
        start_idx = 0
        end_idx = 0

        for i in self.ENV:
            if i != 2019:
                start_idx = end_idx
                end_idx = start_idx + len(self.datasets[i][self.mode])
            elif i == 2019:
                start_idx = 0
                end_idx = len(self.datasets[i][self.mode])
            self.task_idxs[i] = {}
            self.task_idxs[i] = [start_idx, end_idx]

        self.datasets_copy = deepcopy(self.datasets)

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode] = pd.concat([self.datasets[time][self.mode], self.datasets[prev_time][self.mode]])
        self.datasets[time][self.mode].reset_index()
        if data_del:
            del self.datasets[time - 1]

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        if time >= K:
            last_K_timesteps = [self.datasets_copy[time - i][self.mode] for i in range(1, K + 1)]
            self.datasets[time][self.mode] = pd.concat(last_K_timesteps)
            del self.datasets[time - 1][self.mode]
        else:
            self.update_historical(time)

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'drug'


class TdcDtiDg(TdcDtiDgBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        d = self.datasets[self.current_time][self.mode].iloc[index].Drug_Enc
        t = self.datasets[self.current_time][self.mode].iloc[index].Target_Enc

        d = drug_2_embed(d)
        t = protein_2_embed(t)

        y = self.datasets[self.current_time][self.mode].iloc[index].Y
        return (d, t), y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode])


class TdcDtiDgGroup(TdcDtiDgBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.train = True
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            d = drug_2_embed(self.datasets[self.current_time][self.mode].iloc[sel_idx].Drug_Enc)
            t = protein_2_embed(self.datasets[self.current_time][self.mode].iloc[sel_idx].Target_Enc)
            y = self.datasets[self.current_time][self.mode].iloc[sel_idx].Y
            g = torch.LongTensor([groupid])

            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            del groupid

            return (d, t), y, g

        else:
            d = drug_2_embed(self.datasets[self.current_time][self.mode].iloc[index].Drug_Enc)
            t = protein_2_embed(self.datasets[self.current_time][self.mode].iloc[index].Target_Enc)
            y = self.datasets[self.current_time][self.mode].iloc[index].Y

            return (d, t), y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode])


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
              'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100


def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
    else:
        temp = temp[:MAX_SEQ_PROTEIN]
    return temp


def trans_drug(x):
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG - len(temp))
    else:
        temp = temp[:MAX_SEQ_DRUG]
    return temp


def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to drug_preprocessed_{args.reduced_train_prop}.pkl')

    orig_data_file = os.path.join(args.data_dir, f'drug_preprocessed.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = [year for year in list(range(2013, 2021))]
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    start_idx = 0
    end_idx = 0
    task_idxs = {}
    for year in years:
        train_data = dataset[year][0]
        if year != 2019:
            start_idx = end_idx
            end_idx = start_idx + len(train_data)
        elif year == 2019:
            start_idx = 0
            end_idx = len(train_data)
        task_idxs[year] = [start_idx, end_idx]

        num_train_samples = len(train_data)
        reduced_num_train_samples = int(train_fraction * num_train_samples)

        new_train_data = train_data.loc[:reduced_num_train_samples, :].reset_index(drop=True)
        dataset[year][0] = new_train_data

    preprocessed_data_file = os.path.join(args.data_dir, f'drug_preprocessed_{args.reduced_train_prop}.pkl')
    with open(preprocessed_data_file, 'wb') as f:
        pickle.dump(dataset, f)

def preprocess_orig(args):
    group = BenchmarkGroup(name='DTI_DG_Group', path=os.path.join(args.data_dir, 'TDC_OOD'))
    benchmark = group.get('BindingDB_Patent')
    train_val, test, name = benchmark['train_val'], benchmark['test'], benchmark['name']

    unique_drug = pd.Series(train_val['Drug'].unique()).apply(trans_drug)
    unique_dict_drug = dict(zip(train_val['Drug'].unique(), unique_drug))
    train_val['Drug_Enc'] = [unique_dict_drug[str(i)] for i in train_val['Drug']]

    unique_target = pd.Series(train_val['Target'].unique()).apply(trans_protein)
    unique_dict_target = dict(zip(train_val['Target'].unique(), unique_target))
    train_val['Target_Enc'] = [unique_dict_target[str(i)] for i in train_val['Target']]

    unique_drug = pd.Series(test['Drug'].unique()).apply(trans_drug)
    unique_dict_drug = dict(zip(test['Drug'].unique(), unique_drug))
    test['Drug_Enc'] = [unique_dict_drug[str(i)] for i in test['Drug']]

    unique_target = pd.Series(test['Target'].unique()).apply(trans_protein)
    unique_dict_target = dict(zip(test['Target'].unique(), unique_target))
    test['Target_Enc'] = [unique_dict_target[str(i)] for i in test['Target']]

    datasets = {}
    task_idxs = {}
    ENV = [year for year in list(range(2013, 2021))]
    start_idx = 0
    end_idx = 0
    for year in ENV:
        datasets[year] = {}
        if year < 2019:
            df_data = train_val[train_val.Year == year]
        else:
            df_data = test[test.Year == year]

        if year != 2019:
            start_idx = end_idx
            end_idx = start_idx + len(df_data)
        elif year == 2019:
            start_idx = 0
            end_idx = len(df_data)
        task_idxs[year] = [start_idx, end_idx]

        num_samples = len(df_data)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(start_idx, end_idx))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[year][0] = df_data.loc[idxs[:num_train_samples], :].reset_index()
        datasets[year][1] = df_data.loc[idxs[num_train_samples:], :].reset_index()
        datasets[year][2] = df_data.reset_index()

    with open('drug_preprocessed.pkl', 'wb') as f:
        pickle.dump(datasets, f)

def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'drug_preprocessed.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'drug_preprocessed_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)

enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))


def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T


def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray().T
