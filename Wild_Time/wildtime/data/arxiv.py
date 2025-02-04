import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import initialize_distilbert_transform, download_detection

MAX_TOKEN_LENGTH = 300
RAW_DATA_FILE = 'arxiv-metadata-oai-snapshot.json'
ID_HELD_OUT = 0.1

class ArXivBase(Dataset):
    def __init__(self, args):
        super().__init__()
        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}.pkl'
        else:
            self.data_file = f'{str(self)}_{args.reduced_train_prop}.pkl'
        download_detection(args.data_dir, self.data_file)
        preprocess(args)
        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.args = args
        self.ENV = [year for year in range(2007, 2023)]
        self.num_tasks = len(self.ENV)
        self.num_classes = 172
        self.mini_batch_size = args.mini_batch_size
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 0

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0

        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['category'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store class id list
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(np.array(self.datasets[year][self.mode]['category']) == classid)[0]
                self.class_id_list[classid][year] = sel_idx
            print(f'Year {str(year)} loaded')

            # Store input dim
            num_examples = len(self.datasets[year][self.mode]['category'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            if args.method in ['erm']:
                self.input_dim.append(cumulative_batch_size)
            else:
                self.input_dim.append(min(self.mini_batch_size, num_examples))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['title'] = np.concatenate(
            (self.datasets[time][self.mode]['title'], self.datasets[prev_time][self.mode]['title']), axis=0)
        self.datasets[time][self.mode]['category'] = np.concatenate(
            (self.datasets[time][self.mode]['category'], self.datasets[prev_time][self.mode]['category']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['category'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K]
            self.datasets[time][self.mode]['title'] = np.concatenate(
                (self.datasets[time][self.mode]['title'], self.datasets[prev_time][self.mode]['title'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['category'] = np.concatenate(
                (self.datasets[time][self.mode]['category'], self.datasets[prev_time][self.mode]['category'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['category'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        if len(idx_all) == 0:
            return None, None
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)[0]
        title = self.datasets[time_idx][self.mode]['title'][sel_idx]
        category = self.datasets[time_idx][self.mode]['category'][sel_idx]

        x = self.transform(text=title).unsqueeze(0).cuda()
        y = torch.LongTensor([category]).cuda()

        return x, y

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'arxiv'


class ArXiv(ArXivBase):
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

        title = self.datasets[self.current_time][self.mode]['title'][index]
        category = self.datasets[self.current_time][self.mode]['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])


class ArXivGroup(ArXivBase):
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

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            title = self.datasets[self.current_time][self.mode]['title'][sel_idx]
            category = self.datasets[self.current_time][self.mode]['category'][sel_idx]
            x = self.transform(text=title)
            y = torch.LongTensor([category])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            del title
            del category

            return x, y, group_tensor

        else:
            title = self.datasets[self.current_time][self.mode]['title'][index]
            category = self.datasets[self.current_time][self.mode]['category'][index]

            x = self.transform(text=title)
            y = torch.LongTensor([category])

            del title
            del category

            return x, y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])


def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train dataset and saving to arxiv_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'arxiv.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_titles = dataset[year][0]['title']
        train_categories = dataset[year][0]['category']

        num_train_samples = len(train_categories)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_titles = np.array(train_titles)[train_idxs]
        new_train_categories = np.array(train_categories)[train_idxs]
        dataset[year][0]['title'] = np.stack(new_train_titles, axis=0)
        dataset[year][0]['category'] = np.array(new_train_categories)

    preprocessed_data_file = os.path.join(args.data_dir, f'arxiv_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def preprocess_orig(args):
    data_file = os.path.join(args.data_dir, RAW_DATA_FILE)
    if not os.path.isfile(data_file):
        raise ValueError(f'{RAW_DATA_FILE} is not in the data directory {args.data_dir}!')

    base_df = pd.read_json(data_file, lines=True)
    base_df['category'] = base_df.categories.str.split().str.get(0)
    base_df['update_date'] = pd.to_datetime(base_df.update_date)
    base_df = base_df.sort_values(by=['update_date'])
    df_years = base_df.groupby(pd.Grouper(key='update_date', freq='Y'))
    dfs = [group for _, group in df_years]

    years = sorted(pd.unique(pd.DatetimeIndex(base_df['update_date']).year).tolist())
    categories = sorted(pd.unique(base_df['category']).tolist())
    categories_to_classids = {category: classid for classid, category in enumerate(categories)}

    dataset = {}
    for i, year in enumerate(years):
        dataset[year] = {}
        df_year = dfs[i]
        titles = df_year['title'].str.lower().tolist()
        categories = [categories_to_classids[category] for category in df_year['category']]
        num_samples = len(categories)
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        train_idxs = idxs[:num_train_images].astype(int)
        test_idxs = idxs[num_train_images + 1:].astype(int)
        titles_train = np.array(titles)[train_idxs]
        categories_train = np.array(categories)[train_idxs]
        titles_test_id = np.array(titles)[test_idxs]
        categories_test_id = np.array(categories)[test_idxs]

        dataset[year][0] = {}
        dataset[year][0]['title'] = titles_train
        dataset[year][0]['category'] = categories_train
        dataset[year][1] = {}
        dataset[year][1]['title'] = titles_test_id
        dataset[year][1]['category'] = categories_test_id
        dataset[year][2] = {}
        dataset[year][2]['title'] = titles
        dataset[year][2]['category'] = categories

    preprocessed_data_path = os.path.join(args.data_dir, 'arxiv.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'arxiv.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'arxiv_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)

