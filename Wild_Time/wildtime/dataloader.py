scheduler = None
is_group_datasets = ['coral', 'groupdro', 'irm']

def _yearbook_init(args, is_group_data):
    if is_group_data:
        from .data.yearbook import YearbookGroup
        dataset = YearbookGroup(args)
    else:
        from .data.yearbook import Yearbook
        dataset = Yearbook(args)
    return dataset


def _fmow_init(args, is_group_data):
    if is_group_data:
        from .data.fmow import FMoWGroup
        dataset = FMoWGroup(args)
    else:
        from .data.fmow import FMoW
        dataset = FMoW(args)
    return dataset


def _drug_init(args, is_group_data):
    if is_group_data:
        from .data.drug import TdcDtiDgGroup
        dataset = TdcDtiDgGroup(args)
    else:
        from .data.drug import TdcDtiDg
        dataset = TdcDtiDg(args)
    return dataset

def _mimic_init(args, is_group_data):
    if is_group_data:
        from .data.mimic import MIMICGroup
        dataset = MIMICGroup(args)
    else:
        from .data.mimic import MIMIC
        dataset = MIMIC(args)
    return dataset

def _arxiv_init(args, is_group_data):
    if is_group_data:
        from .data.arxiv import ArXivGroup
        dataset = ArXivGroup(args)
    else:
        from .data.arxiv import ArXiv
        dataset = ArXiv(args)
    return dataset

def _huffpost_init(args, is_group_data):
    if is_group_data:
        from .data.huffpost import HuffPostGroup
        dataset = HuffPostGroup(args)
    else:
        from .data.huffpost import HuffPost
        dataset = HuffPost(args)
    return dataset


def getdata(args, is_group_data = False):
    dataset_name = args.dataset
    if dataset_name == 'arxiv' : return _arxiv_init(args, is_group_data)
    if dataset_name == 'drug': return _drug_init(args, is_group_data)
    if dataset_name == 'fmow': return _fmow_init(args, is_group_data)
    if dataset_name == 'huffpost': return _huffpost_init(args, is_group_data)
    if dataset_name == 'mimic': return _mimic_init(args, is_group_data)
    if dataset_name == 'yearbook': return _yearbook_init(args, is_group_data)
