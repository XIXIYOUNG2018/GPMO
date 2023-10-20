# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
import pandas as pd

import torch
import torch.utils.data as tud
from torch.autograd import Variable
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import configuration.config_default as cfgd
from models.transformer.module.subsequent_mask import subsequent_mask
import random


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing
    Source_Mol_ID,Target_Mol_ID,Source_Mol,Target_Mol,
    Source_Mol_LogD,Target_Mol_LogD,Delta_LogD,
    Source_Mol_Solubility,Target_Mol_Solubility,Delta_Solubility,
    Source_Mol_Clint,Target_Mol_Clint,Delta_Clint,
    Transformation,Core"""
    def __init__(self, data, vocabulary, tokenizer, prediction_mode=False):
        """

        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        :param prediction_mode: if use target smiles or not (training or test)
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data
        self._prediction_mode = prediction_mode
        self.task_percent=0.5



    def __getitem__(self, i):
        """
        Tokenize and encode source smile and/or target smile (if prediction_mode is True)
        :param i:
        :return:
        """
        row = self._data.iloc[i]
        #print(row)

        # tokenize and encode source smiles
        #生成的pandas中SMILES1是任意一种smiles，SMILES0是canonical smiles
        # if random.random() < self.task_percent: #翻译任务  
            
        source_smi = row['SMILES1']
        source_tokens = []
        source_tokens.extend(self._tokenizer.tokenize(source_smi))
        source_encoded = self._vocabulary.encode(source_tokens)

    # tokenize and encode target smiles if it is for training instead of evaluation
        if not self._prediction_mode:
            target_smi = row['SMILES0']
            target_tokens = self._tokenizer.tokenize(target_smi)
            target_encoded = self._vocabulary.encode(target_tokens)
            return torch.tensor(source_encoded, dtype=torch.long), torch.tensor(
                target_encoded, dtype=torch.long), row
        else:
            return torch.tensor(source_encoded, dtype=torch.long),  row     
            
        # else:  #掩码任务
        #     target_smi = row['SMILES1']
        #     target_tokens = self._tokenizer.tokenize(target_smi)
        #     target_encoded = self._vocabulary.encode(target_tokens)
        #     source_encoded=my_torch_mask_tokens(torch.tensor(target_encoded, dtype=torch.long))
        #     if not self._prediction_mode:
        #         return torch.tensor(source_encoded, dtype=torch.long), torch.tensor(
        #             target_encoded, dtype=torch.long), row
        #     else:
        #         return torch.tensor(source_encoded, dtype=torch.long),  row     



    def __len__(self):
        return len(self._data)

    @classmethod
    def collate_fn(cls, data_all):
        # sort based on source sequence's length
        #data_all.sort(key=lambda x: len(x[0]), reverse=True)
        #print(data_all)
        is_prediction_mode = True if len(data_all[0]) == 2 else False
        if is_prediction_mode:
            source_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)
        else:
            source_encoded, target_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)
        # maximum length of source sequences
        max_length_source = max([seq.size(0) for seq in source_encoded])
        # padded source sequences with zeroes
        collated_arr_source = torch.zeros(len(source_encoded), max_length_source, dtype=torch.long)
        for i, seq in enumerate(source_encoded):
            collated_arr_source[i, :seq.size(0)] = seq
        # length of each source sequence
        source_length = [seq.size(0) for seq in source_encoded]
        source_length = torch.tensor(source_length)
        # mask of source seqs
        src_mask = (collated_arr_source !=0).unsqueeze(-2)

            # target seq
        if not is_prediction_mode:
            max_length_target = max([seq.size(0) for seq in target_encoded])
            collated_arr_target = torch.zeros(len(target_encoded), max_length_target, dtype=torch.long)
            for i, seq in enumerate(target_encoded):
                collated_arr_target[i, :seq.size(0)] = seq

            trg_mask = (collated_arr_target != 0).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask))
            #trg_mask1=trg_mask.clone()#用作在使用mlm任务时给掩码的collated_arr_source
            trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token
            if random.random() < 0.5:
                collated_arr_source=my_torch_mask_tokens(collated_arr_target)
                src_mask= (collated_arr_source !=0).unsqueeze(-2)
        else:
            trg_mask = None
            max_length_target = None
            collated_arr_target = None
        #print(collated_arr_source.shape,src_mask.shape,collated_arr_target.shape,trg_mask.shape)

        #print(collated_arr_source.shape,src_mask.shape,collated_arr_target.shape,trg_mask.shape)      
        
        return collated_arr_source, source_length, collated_arr_target, src_mask, trg_mask, max_length_target, data

def my_torch_mask_tokens(inputs,
                         special_tokens_mask: Optional[Any] = None,
                         mlm_probability=0.15) -> Tuple[Any, Any]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # labels = inputs.clone()
    input_shape = inputs.shape
    #print(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(input_shape, mlm_probability)
    #print(special_tokens_mask)
    if special_tokens_mask is None:
        special_tokens_mask =  [get_special_tokens_mask(val) for val in inputs]
        special_tokens_mask=[t.tolist() for t in special_tokens_mask] 
        special_tokens_mask=torch.Tensor(special_tokens_mask)
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()
    #print((special_tokens_mask[0]==True).sum(),(inputs[0]==0).sum())
    #print(special_tokens_mask[1],inputs[1])
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        input_shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 0
    # print(torch.bernoulli(torch.full(input_shape, 0.8)).bool())
    # print(masked_indices)
    # 10% of the time, we replace masked input tokens with random word
    #print(inputs[0])
    indices_random = torch.bernoulli(torch.full(
        input_shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.zeros(
         input_shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    #print(inputs[0],indices_random)
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # return inputs, labels
    return inputs
def get_special_tokens_mask(val):
    #将数据中为special tokens的地方都设置为0
    one = torch.ones_like(val)
    zero = torch.zeros_like(val)
    return torch.where(val > 2, zero, one)
