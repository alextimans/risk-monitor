import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import initialize_distilbert_transform, download_detection

PREPROCESSED_FILE = 'huffpost.pkl'
MAX_TOKEN_LENGTH = 300
RAW_DATA_FILE = 'News_Category_Dataset_v2.json'
ID_HELD_OUT = 0.1

class HuffPostBase(Dataset):
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
        self.ENV = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
        self.num_classes = 11 # 41 if we don't remove classes
        self.num_tasks = len(self.ENV)
        self.current_time = 0
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

        # total_samples = 0
        # for i in self.ENV:
        #     total_samples += len(self.datasets[i][2]['category'])
        # print('total', total_samples)

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['headline'] = np.concatenate(
            (self.datasets[time][self.mode]['headline'], self.datasets[prev_time][self.mode]['headline']), axis=0)
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
            self.datasets[time][self.mode]['headline'] = np.concatenate(
                (self.datasets[time][self.mode]['headline'], self.datasets[prev_time][self.mode]['headline'][:-last_K_num_samples]), axis=0)
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
        headline = self.datasets[time_idx][self.mode]['headline'][sel_idx]
        category = self.datasets[time_idx][self.mode]['category'][sel_idx]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x.unsqueeze(0).cuda(), y.cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'huffpost'


class HuffPost(HuffPostBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        headline = self.datasets[self.current_time][self.mode]['headline'][index]
        category = self.datasets[self.current_time][self.mode]['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])


class HuffPostGroup(HuffPostBase):
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
            headline = self.datasets[self.current_time][self.mode]['headline'][sel_idx]
            category = self.datasets[self.current_time][self.mode]['category'][sel_idx]
            x = self.transform(text=headline)
            y = torch.LongTensor([category])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return x, y, group_tensor

        else:
            headline = self.datasets[self.current_time][self.mode]['headline'][index]
            category = self.datasets[self.current_time][self.mode]['category'][index]

            x = self.transform(text=headline)
            y = torch.LongTensor([category])

            return x, y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])


"""
News Categories to IDs:
    {'BLACK VOICES': 0, 'BUSINESS': 1, 'COMEDY': 2, 'CRIME': 3, 
    'ENTERTAINMENT': 4, 'IMPACT': 5, 'QUEER VOICES': 6, 'SCIENCE': 7, 
    'SPORTS': 8, 'TECH': 9, 'TRAVEL': 10}
"""

def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to huffpost_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'huffpost.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_headlines = dataset[year][0]['headline']
        train_categories = dataset[year][0]['category']

        num_train_samples = len(train_categories)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_headlines = np.array(train_headlines)[train_idxs]
        new_train_categories = np.array(train_categories)[train_idxs]
        dataset[year][0]['title'] = np.stack(new_train_headlines, axis=0)
        dataset[year][0]['category'] = np.array(new_train_categories)

    preprocessed_data_file = os.path.join(args.data_dir, f'huffpost_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def preprocess_orig(args):
    raw_data_path = os.path.join(args.data_dir, RAW_DATA_FILE)
    if not os.path.isfile(raw_data_path):
        raise ValueError(f'{RAW_DATA_FILE} is not in the data directory {args.data_dir}!')

    # Load data frame from json file, group by year
    base_df = pd.read_json(raw_data_path, lines=True)
    base_df = base_df.sort_values(by=['date'])
    df_years = base_df.groupby(pd.Grouper(key='date', freq='Y'))
    dfs = [group for _, group in df_years]
    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]

    # Identify class ids that appear in all years 2012 - 2018
    categories_to_classids = {category: classid for classid, category in
                              enumerate(sorted(base_df['category'].unique()))}
    classids_to_categories = {v: k for k, v in categories_to_classids.items()}
    classids = []
    num_classes = len(categories_to_classids.values())
    for classid in range(num_classes):
        class_count = 0
        for i, year in enumerate(years):
            year_classids = [categories_to_classids[category] for category in dfs[i]['category']]
            if classid in year_classids:
                class_count += 1
        if class_count == len(years):
            classids.append(classid)

    # Re-index the class ids that appear in all years 2012 - 2018 and store them
    classids_to_categories = {i: classids_to_categories[classid] for i, classid in enumerate(classids)}
    categories_to_classids = {v: k for k, v in classids_to_categories.items()}

    dataset = {}
    for i, year in enumerate(years):
        # Store news headlines and category labels
        dataset[year] = {}
        df_year = dfs[i]
        df_year = df_year[df_year['category'].isin(categories_to_classids.keys())]
        headlines = df_year['headline'].str.lower().tolist()
        categories = [categories_to_classids[category] for category in df_year['category']]

        num_samples = len(categories)
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        train_idxs = idxs[:num_train_images].astype(int)
        test_idxs = idxs[num_train_images + 1:].astype(int)
        headlines_train = np.array(headlines)[train_idxs]
        categories_train = np.array(categories)[train_idxs]
        headlines_test_id = np.array(headlines)[test_idxs]
        categories_test_id = np.array(categories)[test_idxs]

        dataset[year][0] = {}
        dataset[year][0]['headline'] = headlines_train
        dataset[year][0]['category'] = categories_train
        dataset[year][1] = {}
        dataset[year][1]['headline'] = headlines_test_id
        dataset[year][1]['category'] = categories_test_id
        dataset[year][2] = {}
        dataset[year][2]['headline'] = headlines
        dataset[year][2]['category'] = categories

    preprocessed_data_path = os.path.join(args.data_dir, 'huffpost.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'huffpost.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'huffpost_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)

