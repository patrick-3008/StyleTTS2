#coding: utf-8
import random
import numpy as np
import random

import torch
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

from char_indexer import VanillaCharacterIndexer, BertCharacterIndexer

np.random.seed(1)
random.seed(1)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 data_augmentation=False,
                 validation=False,
                 min_length=50,
                 ):

        self.dataset = dataset
        self.char_indexer = VanillaCharacterIndexer()
        self.bert_char_indexer = BertCharacterIndexer()

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):        
        sample = self.dataset[idx]
        assert sample['sampling_rate'] == 24000

        # Load original sample
        wave, text_tensor, bert_text_tensor, speaker_id = self._load_tensor(sample)
        
        # Process original audio into mel spectrogram
        mel_tensor = preprocess(wave).squeeze()
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # Get reference sample from the same speaker
        ref_idx = random.randint(0, len(self.dataset) - 1)
        same_speaker_sample = self.dataset[ref_idx]
        ref_mel_tensor, _ = self._load_data(same_speaker_sample)
        
        return speaker_id, acoustic_feature, text_tensor, bert_text_tensor, ref_mel_tensor, wave

    def _load_tensor(self, sample):
        text = sample['phonemes']
        speaker_id = sample.get('speaker_id', 0)

        wave = np.array(sample['audio'][0])
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        char_idx = self.char_indexer(text)
        bert_char_idx = self.bert_char_indexer(text)
        
        char_idx.insert(0, 0); bert_char_idx.insert(0, 0)
        char_idx.append(0); bert_char_idx.append(0)
        
        char_idx = torch.LongTensor(char_idx)
        bert_char_idx = torch.LongTensor(bert_char_idx)

        return wave, char_idx, bert_char_idx, speaker_id

    def _load_data(self, data):
        wave, _, _, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        bert_texts = torch.zeros((batch_size, max_text_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, bert_text, ref_mel, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            bert_texts[bid, :text_size] = bert_text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            waves[bid] = wave

        return waves, texts, bert_texts, input_lengths, mels, output_lengths, ref_mels

def build_dataloader(dataset, min_length, batch_size, num_workers, device, validation=False, collate_config={}, dataset_config={}, **kwargs):
    dataset = FilePathDataset(dataset,  min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

