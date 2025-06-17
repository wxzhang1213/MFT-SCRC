import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.prepare_data import prepare_data_instances

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        data_path = self.args.dataPath + self.args.datasetName + '/unaligned_50.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode + '_labels'].astype(np.float32)
        }
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        text_lengths = np.argmin(self.text[:, 1, :], axis=1).astype(np.int16).tolist()
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        input_masks = []
        for length in text_lengths:
            mask = np.array([1] * length + [0] * (self.audio.shape[1] - length))
            input_masks.append(mask)
        self.input_mask = np.array(input_masks)

    def __init_mosei(self):
        return self.__init_mosi()

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'audio_lengths': self.audio_lengths[index],
            'vision': torch.Tensor(self.vision[index]),
            'vision_lengths': self.vision_lengths[index],
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader


def MMDataLoader_1(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    train_graph = prepare_data_instances(datasets['train'], args)
    valid_graph = prepare_data_instances(datasets['valid'], args)
    test_graph = prepare_data_instances(datasets['test'], args)

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader, train_graph, valid_graph, test_graph