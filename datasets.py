import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

class ProteinDataset(Dataset):
  AA_LIST = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
  NUM_AA = len(AA_LIST)
  IDX_DICT = {x:i for i, x in enumerate(AA_LIST)}
  NUM_METALS = 29

  def __init__(self, data_path, filter_path=None, max_len=None, pos_only=False, neighbourhood=None):
    self.max_len = max_len
    self.neighbourhood = neighbourhood
    self.df = pd.read_csv(data_path, sep='\t')

    if self.max_len:
      self.df = self.df[self.df["Length"] <= self.max_len]
    
    self.df["Indices"].fillna("", inplace=True)
    self.df = self.df.set_index("Accession")

    if filter_path:
      self.filter = pd.read_csv(filter_path, header=None)
      self.df = self.df.loc[np.intersect1d(self.df.index, self.filter[0])]

    if pos_only:
      self.df = self.df[self.df["Indices"]!=""]

    self.unique_accs = self.df.index.unique()
    assert len(self.df) == len(self.unique_accs)

    self.acc_to_xy = {}
    self.process_data()


  def process_data(self):
    for acc in self.df.index:
      data = self.df.loc[acc]

      x = data["Sequence"]
      seq_len = int(data["Length"])
      idxbuffer = data["Indices"]

      x = self.one_hot_encode(x)

      if idxbuffer == "":
        y = np.zeros((seq_len, self.NUM_METALS), dtype='b')

      else:
        y = self.decode_buffer(idxbuffer, seq_len)
        if self.neighbourhood is not None:
          binding_cols = np.nonzero(y)[0]
          first, last = binding_cols[0], binding_cols[-1]
          x = x[max(0, first-self.neighbourhood):last+self.neighbourhood+1]
          y = y[max(0, first-self.neighbourhood):last+self.neighbourhood+1,:]
          seq_len = y.shape[0]
      
      self.acc_to_xy[acc] = (x, y, seq_len)


  def one_hot_encode(self, x):
    x = pd.Series(list(x))
    x = x.map(self.IDX_DICT).fillna(0).astype(int)
    x = x.to_numpy() 
    return (np.arange(self.NUM_AA-1) == x[...,None]-1).astype('b')


  def decode_buffer(self, idxbuffer, seq_len):
    idxs = np.frombuffer(literal_eval(idxbuffer), dtype='int64')
    y = np.zeros(seq_len*self.NUM_METALS, dtype='b')
    y[idxs] = 1
    y = y.reshape((seq_len, self.NUM_METALS))
    return y


  def __getitem__(self, idx):
    acc = self.unique_accs[idx]
    x, y, seq_len = self.acc_to_xy[acc]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y, seq_len, acc


  def __len__(self):
    return self.unique_accs.size


def pad_batch_seqs(batch):
  (seqs, targets, lens, accs) = zip(*batch)
  
  padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
  padded_seqs = torch.transpose(padded_seqs, 1, 2) # (batch size, encoding_dim, padded seq. length)
  
  padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=2)
  padded_targets = torch.transpose(padded_targets, 1, 2) # (batch size, num. metals, padded seq. length)
  mask = (padded_targets != 2)
  padded_targets[~mask] = 0

  return padded_seqs, padded_targets, mask



class CARPDataset(Dataset):
  AA_LIST = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
  NUM_AA = len(AA_LIST)
  IDX_DICT = {x:i for i, x in enumerate(AA_LIST)}
  NUM_METALS = 29

  def __init__(self, data_path, filter_path=None, max_len=None):
    self.max_len = max_len
    self.df = pd.read_csv(data_path, sep='\t')

    if self.max_len:
      self.df = self.df[self.df["Length"] <= self.max_len]
    
    self.df["Indices"].fillna("", inplace=True)
    self.df = self.df.set_index("Accession")

    if filter_path:
      self.filter = pd.read_csv(filter_path, header=None)
      self.df = self.df.loc[np.intersect1d(self.df.index, self.filter[0])]

    self.unique_accs = self.df.index.unique()
    assert len(self.df) == len(self.unique_accs)

    self.acc_to_xy = {}
    self.process_data()


  def process_data(self):
    for acc in self.df.index:
      data = self.df.loc[acc]

      x = data["Sequence"]
      seq_len = int(data["Length"])
      idxbuffer = data["Indices"]

      if idxbuffer == "":
        y = np.zeros((seq_len, self.NUM_METALS), dtype='b')

      else:
        y = self.decode_buffer(idxbuffer, seq_len)
      
      self.acc_to_xy[acc] = (x, y, seq_len)



  def decode_buffer(self, idxbuffer, seq_len):
    idxs = np.frombuffer(literal_eval(idxbuffer), dtype='int64')
    y = np.zeros(seq_len*self.NUM_METALS, dtype='b')
    y[idxs] = 1
    y = y.reshape((seq_len, self.NUM_METALS))
    return y


  def __getitem__(self, idx):
    acc = self.unique_accs[idx]
    x, y, seq_len = self.acc_to_xy[acc]
    y = torch.from_numpy(y)

    return [x], y, seq_len, acc

  def __len__(self):
    return self.unique_accs.size


def CARP_batch_seqs(batch, model, collater, device):
  (seqs, targets, lens, accs) = zip(*batch)
  with torch.no_grad():
    x = collater(seqs)[0]
    x = x.to(device)
    rep = model(x)
    padded_seqs = rep["representations"][32]
  
  padded_seqs = torch.transpose(padded_seqs, 1, 2) # (batch size, encoding_dim, padded seq. length)
  
  padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=2)
  padded_targets = torch.transpose(padded_targets, 1, 2) # (batch size, num. metals, padded seq. length)
  mask = (padded_targets != 2)
  padded_targets[~mask] = 0

  return padded_seqs, padded_targets, mask