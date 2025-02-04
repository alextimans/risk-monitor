import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from wilds import get_dataset

from .utils import download_detection

ID_HELD_OUT = 0.1

class FMoWBase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}.pkl'
        else:
            self.data_file = f'{str(self)}_{args.reduced_train_prop}.pkl'

        download_detection(args.data_dir, self.data_file)
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=True)
        self.root = dataset.root

        self.args = args
        self.num_classes = 62
        self.current_time = 0
        self.num_tasks = 17
        self.ENV = [year for year in range(0, self.num_tasks - 1)]
        self.resolution = 224
        self.mode = 0
        self.ssl_training = False

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        start_idx = 0
        for year in sorted(self.datasets.keys()):
            count = len(self.datasets[year][self.mode]['labels'])
            cumulative_batch_size += min(args.mini_batch_size, count)
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(args.mini_batch_size, count), 3, 32, 32))

            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = {}
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[year][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][year] = sel_idx

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'fmow'

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['image_idxs'] = np.concatenate(
            (self.datasets[time][self.mode]['image_idxs'], self.datasets[prev_time][self.mode]['image_idxs']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['image_idxs'] = np.concatenate(
                (self.datasets[time][self.mode]['image_idxs'], self.datasets[prev_time][self.mode]['image_idxs'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.datasets[self.current_time][self.mode]['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = torch.stack([self.transform(self.get_input(idx)) for idx in sel_idx], dim=0)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        return torch.FloatTensor(image).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()


class FMoW(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, idx):
        if self.args.difficulty and self.mode == 0:
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            idx = sel_idx

        image_tensor = self.transform(self.get_input(idx))
        label_tensor = torch.LongTensor([self.datasets[self.current_time][self.mode]['labels'][idx]])

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            tensor_to_PIL = transforms.ToPILImage()
            image_tensor = tensor_to_PIL(image_tensor)
            return image_tensor, label_tensor, ''

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class FMoWGroup(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.train = True
        self.groupnum = 0

    def __getitem__(self, idx):
        if self.mode == 0:
            np.random.seed(idx)
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
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]
            image_tensor = self.transform(self.get_input(sel_idx))
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return image_tensor, label_tensor, group_tensor

        else:
            image_tensor = self.transform(self.get_input(idx))
            label = self.datasets[self.current_time][self.mode]['labels'][idx]
            label_tensor = torch.LongTensor([label])

            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to fmow_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'fmow.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_image_idxs = dataset[year][0]['image_idxs']
        train_labels = dataset[year][0]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_image_idxs = np.array(train_image_idxs)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][0]['image_idxs'] = np.stack(new_train_image_idxs, axis=0)
        dataset[year][0]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(args.data_dir, f'fmow_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def get_image_idxs_and_labels(split: str, data_dir: str):
    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=True)
    split_array = dataset.split_array
    split_dict = dataset.split_dict

    y_array = []

    split_mask = split_array == split_dict[split]
    split_idx = np.where(split_mask)[0]
    y_array.append(dataset.y_array[split_idx])
    years = dataset.metadata_array[split_idx, 1]
    split_unique_years = torch.unique(years).detach().numpy().tolist()

    image_idxs = defaultdict(list)
    labels = defaultdict(list)
    for year in split_unique_years:
        image_idxs[year].append(dataset.full_idxs[split_idx][torch.where(years == year)])
        labels[year].append(dataset.y_array[split_idx][torch.where(years == year)])

    return image_idxs, labels


def get_train_test_split(image_idxs_year, labels_year):
    num_samples = len(labels_year)
    num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
    seed_ = np.random.get_state()
    np.random.seed(0)
    idxs = np.random.permutation(np.arange(num_samples))
    np.random.set_state(seed_)
    train_image_idxs = image_idxs_year[idxs[:num_train_samples]]
    train_labels = labels_year[idxs[:num_train_samples]]
    test_image_idxs = image_idxs_year[idxs[num_train_samples:]]
    test_labels = labels_year[idxs[num_train_samples:]]

    return train_image_idxs, train_labels, test_image_idxs, test_labels


def preprocess_orig(args):
    datasets = {}
    train_image_idxs, train_labels = get_image_idxs_and_labels('train', args.data_dir)
    val_image_idxs, val_labels = get_image_idxs_and_labels('id_val', args.data_dir)
    test_id_image_idxs, test_id_labels = get_image_idxs_and_labels('id_test', args.data_dir)

    # ID Years 2002 - 2013
    for year in range(0, 11):
        datasets[year] = {}
        datasets[year][0] = {}
        datasets[year][1] = {}
        datasets[year][2] = {}

        datasets[year][0]['image_idxs'] = np.concatenate((train_image_idxs[year][0], val_image_idxs[year][0]), axis=0)
        datasets[year][0]['labels'] = np.concatenate((train_labels[year][0], val_labels[year][0]), axis=0)
        datasets[year][1]['image_idxs'] = np.array(test_id_image_idxs[year][0])
        datasets[year][1]['labels'] = np.array(test_id_labels[year][0])
        datasets[year][2]['image_idxs'] = np.concatenate((datasets[year][0]['image_idxs'], datasets[year][1]['image_idxs']), axis=0)
        datasets[year][2]['labels'] = np.concatenate((datasets[year][0]['labels'], datasets[year][1]['labels']), axis=0)
    del train_image_idxs, train_labels, val_image_idxs, val_labels, test_id_image_idxs, test_id_labels

    # Intermediate Years 2013 - 2016
    val_ood_image_idxs, val_ood_labels = get_image_idxs_and_labels('val', args.data_dir)
    for year in range(11, 14):
        datasets[year] = {}
        datasets[year][0] = {}
        datasets[year][1] = {}
        datasets[year][2] = {}

        train_image_idxs, train_labels, test_image_idxs, test_labels = get_train_test_split(val_ood_image_idxs[year][0], val_ood_labels[year][0])
        datasets[year][0]['image_idxs'] = train_image_idxs
        datasets[year][0]['labels'] = train_labels
        datasets[year][1]['image_idxs'] = test_image_idxs
        datasets[year][1]['labels'] = test_labels
        datasets[year][2]['image_idxs'] = val_ood_image_idxs[year][0]
        datasets[year][2]['labels'] = val_ood_labels[year][0]
        del train_image_idxs, train_labels, test_image_idxs, test_labels
    del val_ood_image_idxs, val_ood_labels

    # OOD Years 2016 - 2018
    test_ood_image_idxs, test_ood_labels = get_image_idxs_and_labels('test', args.data_dir)
    for year in range(14, 16):
        datasets[year] = {}
        datasets[year][0] = {}
        datasets[year][1] = {}
        datasets[year][2] = {}

        train_image_idxs, train_labels, test_image_idxs, test_labels = get_train_test_split(test_ood_image_idxs[year][0], test_ood_labels[year][0])
        datasets[year][0]['image_idxs'] = train_image_idxs
        datasets[year][0]['labels'] = train_labels
        datasets[year][1]['image_idxs'] = test_image_idxs
        datasets[year][1]['labels'] = test_labels
        datasets[year][2]['image_idxs'] = test_ood_image_idxs[year][0]
        datasets[year][2]['labels'] = test_ood_labels[year][0]
        del train_image_idxs, train_labels, test_image_idxs, test_labels
    del test_ood_image_idxs, test_ood_labels

    preprocessed_data_path = os.path.join(args.data_dir, 'fmow.pkl')
    pickle.dump(datasets, open(preprocessed_data_path, 'wb'))

def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'fmow.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'fmow_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)
