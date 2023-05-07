import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, multilabel_confusion_matrix
from pathlib import Path
import pickle
from sequence_models.pretrained import load_model_and_alphabet
from functools import partial
from early_stopping import EarlyStopping
from cnn_models import CARP_CNN
from datasets import CARPDataset, CARP_batch_seqs

# dataset filepaths
CARP_IN = "carp-weights"
IN = "train-test/splits"

BYTES_TRAIN_SET_FILEPATH = f"{IN}/train.tsv"
BYTES_VAL_SET_FILEPATH = f"{IN}/validation.tsv"
BYTES_TEST_SET_FILEPATH = f"{IN}/test.tsv"
BYTES_TRAINVAL_SET_FILEPATH = f"{IN}/trainval.tsv"

BYTES_BAL_TRAIN_SET_FILEPATH = f"{IN}/balanced_train.tsv"
BYTES_BAL_VAL_SET_FILEPATH = f"{IN}/balanced_validation.tsv"
BYTES_BAL_TRAINVAL_SET_FILEPATH = f"{IN}/balanced_trainval.tsv"
BYTES_BAL_TEST_SET_FILEPATH = f"{IN}/balanced_test.tsv"

SEQ_FILTER_40_TRAINVAL_FILEPATH = f"{IN}/trainval_filtered40.txt"
SEQ_FILTER_40_TEST_FILEPATH = f"{IN}/test_filtered40.txt"


def clear_gpu():
  with torch.no_grad():
    torch.cuda.empty_cache()

"""
trains a network under the cross-validation procedure
"""
def train_CNN(model_class, criterion, loss_name, train_data, val_data, carp_path, encoding_dim,
              n_classes=29, window_len=3, batch_size=64, learning_rate=5e-4, weight_decay=0,
              accum_iter=1, subset_dir="", load_last_model_state=False, early_stop_patience=5, epoch_cap=50):
  
  model_name = str(model_class.__name__)
  model_dir = f"{model_name}_{window_len}W_{loss_name}"
  final_dir = f"MODEL_EXPERIMENTS/{subset_dir}/{model_dir}"
  Path(final_dir).mkdir(parents=True, exist_ok=load_last_model_state)

  train_loop(model_class, criterion, train_data, val_data, final_dir, carp_path, encoding_dim,
              n_classes=n_classes, window_len=window_len, batch_size=batch_size, 
              learning_rate = learning_rate, load_last_model_state=load_last_model_state, weight_decay=weight_decay,
              early_stop_patience=early_stop_patience, epoch_cap=epoch_cap, accum_iter=accum_iter)
  try:
    print("evaluating on validation set")
    calculate_metrics(model_class, final_dir, val_data, thresholds = 0.5, eval_set_dir="validation", zero_div=0)
    print("evaluating on training set")
    calculate_metrics(model_class, final_dir, train_data, thresholds = 0.5, eval_set_dir="train", zero_div=0)
  except Exception as e:
    print(e) 
  

"""cross-validation training loop"""
def train_loop(model_class, criterion, train_data, val_data, final_dir, carp_path, encoding_dim,
               n_classes=29, window_len=3, batch_size=64, learning_rate=1e-3, weight_decay=0,
               load_last_model_state=False, epoch_cap=50, early_stop_patience=None, save_interval = 50, accum_iter=1):
  
  # use GPU if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # load pretrained model weights and sequence collater
  carp_model, carp_collater = load_model_and_alphabet(carp_path)
  carp_model.to(device)
  model = model_class(n_classes, window_len, encoding_dim = encoding_dim)

  # intermediate model weights
  state_path = f"{final_dir}/state_dict.pt"
  # intermediate optimizer state
  opt_state_path = f"{final_dir}/opt_state_dict.pt"
  train_loss_path = f"{final_dir}/running_train_losses"
  val_loss_path = f"{final_dir}/running_val_losses"

  running_train_losses = []
  running_val_losses = []

  if load_last_model_state:
    model.load_state_dict(torch.load(state_path))
    with open(train_loss_path, "rb") as f:
      running_train_losses = pickle.load(f)
    with open(val_loss_path, "rb") as f:
      running_val_losses = pickle.load(f)

  if running_val_losses:
    last_val_loss = running_val_losses[-1]
  else:
    last_val_loss = np.Inf

  model.to(device)

  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                collate_fn = partial(CARP_batch_seqs, model=carp_model, collater=carp_collater, device=device))
  val_dataloader = DataLoader(val_data, batch_size=1,
                              collate_fn = partial(CARP_batch_seqs, model=carp_model, collater=carp_collater, device=device))

  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  if load_last_model_state:
    optimizer.load_state_dict(torch.load(opt_state_path))

  # learning rate halves if after 10 consecutive epochs there is no drop in val. loss
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.5,patience=10)
  early_stopping = EarlyStopping(patience=early_stop_patience, model_path=state_path, opt_path=opt_state_path,
                                 verbose=False, last_val_loss=last_val_loss)

  start_epoch = len(running_train_losses)
  for epoch in range(1+start_epoch, epoch_cap+1):
    # EPOCH LOOP START
    train_losses = []
    val_losses = []

    # TRAIN LOOP START
    model.train()
    batch_loss = 0
    for batch_idx, (seqs, targets, mask) in enumerate(train_dataloader):
      clear_gpu()
      seqs, targets = seqs.to(device), targets.to(device)
      mask = mask.to(device)
      preds = model(seqs.type(torch.float)) # (batch size, num. metals, padded seq. length)
      loss = criterion(preds, targets.type(torch.float))
      batch_loss = batch_loss + (loss[mask]).mean()
      loss.detach()
      
      # once losses for accum_iter number of minibatches summed up then do the backward pass
      if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
        batch_loss = batch_loss / accum_iter
        train_losses.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss = 0
    # TRAIN LOOP END

    if (epoch % save_interval == 0):
      chckpnt = f"{final_dir}/{epoch}"
      Path(chckpnt).mkdir(parents=True, exist_ok=load_last_model_state)
      torch.save(model.state_dict(), f"{chckpnt}/state_dict.pt")
      torch.save(optimizer.state_dict(), f"{chckpnt}/opt_state_dict.pt")

    # VALIDATION LOOP START  
    model.eval()
    with torch.no_grad():
      val_loss = 0
      for seqs, targets, _ in val_dataloader: # validation dataloader has batch size 1
        clear_gpu()
        seqs, target = seqs[0], targets[0]
        seqs, target = seqs.to(device), target.to(device)
        preds = model(seqs.type(torch.float)) # (num. metals, seq. length)
        loss = criterion(preds, target.type(torch.float)).mean()
        val_losses.append(loss.item())
        val_loss = val_loss + loss.item()
        loss.detach()
        
    if early_stop_patience:
      scheduler.step(val_loss)

    # VALIDATION LOOP END

    # BACK TO EPOCH LOOP
    train_loss = np.mean(train_losses)
    running_train_losses.append(train_loss)
    val_loss = np.mean(val_losses)
    running_val_losses.append(val_loss)

    print(f"Epoch {epoch}: train loss - {train_loss}, val. loss - {val_loss}")
    
    early_stopping(val_loss, model, optimizer)
    if early_stopping.early_stop:
      print("Early stopping")
      break

    with open(train_loss_path, "wb") as f:
      pickle.dump(running_train_losses, f)
    with open(val_loss_path, "wb") as f:
      pickle.dump(running_val_losses, f)
    # EPOCH LOOP END


"""train a final model on full development set for unseen seq. inference"""
def train_final_CNN(model_class, criterion, loss_name, train_data, test_data, carp_path, encoding_dim,
              n_classes=29, window_len=3, batch_size=64, learning_rate=1e-3, weight_decay=0,
              accum_iter=1, subset_dir="", load_last_model_state=False, epoch_cap=100):
  
  model_name = str(model_class.__name__)
  model_dir = f"{model_name}_{window_len}W_{loss_name}"
  final_dir = f"FINAL_MODEL/{subset_dir}/{model_dir}"
  Path(final_dir).mkdir(parents=True, exist_ok=load_last_model_state)

  final_train_loop(model_class, criterion, train_data, test_data, final_dir, carp_path, encoding_dim,
              n_classes=n_classes, window_len=window_len, batch_size=batch_size, 
              learning_rate = learning_rate, load_last_model_state=load_last_model_state, weight_decay=weight_decay,
               epoch_cap=epoch_cap, accum_iter=accum_iter)
  

"""training loop for final model"""
def final_train_loop(model_class, criterion, train_data, test_data, final_dir, carp_path, encoding_dim,
               n_classes=29, window_len=3, batch_size=64, learning_rate=1e-3, weight_decay=0,
               load_last_model_state=False, epoch_cap=100, save_interval = 25, accum_iter=1):
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  carp_model, carp_collater = load_model_and_alphabet(carp_path)
  carp_model.to(device)
  model = model_class(n_classes, window_len, encoding_dim=encoding_dim)

  state_path = f"{final_dir}/state_dict.pt"
  opt_state_path = f"{final_dir}/opt_state_dict.pt"
  train_loss_path = f"{final_dir}/running_train_losses"
  val_loss_path = f"{final_dir}/running_val_losses"

  running_train_losses = []

  if load_last_model_state:
    model.load_state_dict(torch.load(state_path))
    with open(train_loss_path, "rb") as f:
      running_train_losses = pickle.load(f)

  model.to(device)

  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                collate_fn = partial(CARP_batch_seqs, model=carp_model, collater=carp_collater, device=device))

  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  if load_last_model_state:
    optimizer.load_state_dict(torch.load(opt_state_path))

  start_epoch = len(running_train_losses)
  for epoch in range(1+start_epoch, epoch_cap+1):
    # EPOCH LOOP START
    train_losses = []

    # TRAIN LOOP START
    model.train()
    batch_loss = 0
    for batch_idx, (seqs, targets, mask) in enumerate(train_dataloader):
      clear_gpu()
      seqs, targets = seqs.to(device), targets.to(device)
      mask = mask.to(device)
      preds = model(seqs.type(torch.float)) # (batch size, num. metals, padded seq. length)
      loss = criterion(preds, targets.type(torch.float))
      batch_loss = batch_loss + (loss[mask]).mean()
      loss.detach()
      
      # once losses for accum_iter number of minibatches summed up then do the backward pass
      if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
        batch_loss = batch_loss / accum_iter
        train_losses.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss = 0
    # TRAIN LOOP END

    if (epoch % save_interval == 0):
      chckpnt = f"{final_dir}/{epoch}"
      Path(chckpnt).mkdir(parents=True, exist_ok=load_last_model_state)
      torch.save(model.state_dict(), f"{chckpnt}/state_dict.pt")
      torch.save(optimizer.state_dict(), f"{chckpnt}/opt_state_dict.pt")
      try:
        print("evaluating on test set")
        calculate_metrics(model_class, chckpnt, test_data, thresholds = 0.5, eval_set_dir="test", zero_div=0)
      except Exception as e:
        print(e) 

    # BACK TO EPOCH LOOP
    train_loss = np.mean(train_losses)
    running_train_losses.append(train_loss)
    
    print(f"Epoch {epoch}: train loss - {train_loss}")
    
    torch.save(model.state_dict(), state_path)
    torch.save(optimizer.state_dict(), opt_state_path)

    with open(train_loss_path, "wb") as f:
      pickle.dump(running_train_losses, f)
    # EPOCH LOOP END


### METRICS functions ### 
"""calculation of metrics and saves predicted probabilities"""
def calculate_metrics(model_class, final_dir, val_data, carp_path, encoding_dim, 
                      thresholds = 0.5, eval_set_dir="validation", zero_div=0, 
                      n_classes=29, window_len=3):
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  carp_model, carp_collater = load_model_and_alphabet(carp_path)
  carp_model.to(device)

  if torch.is_tensor(thresholds):
    thresholds = thresholds.to(device)
  
  state_path = f"{final_dir}/state_dict.pt"

  threshold_path = f"{final_dir}/{eval_set_dir}"

  Path(threshold_path).mkdir(parents=True, exist_ok=True)

  model = model_class(n_classes=n_classes, window_len=window_len, encoding_dim=encoding_dim)
  model.load_state_dict(torch.load(state_path, map_location=device))
  model.to(device)
  model.eval()

  val_dataloader = DataLoader(val_data,
                              collate_fn = partial(CARP_batch_seqs, model=carp_model, collater=carp_collater, device=device))
  # store big matrix of all label predictions and ground truth across all test examples
  all_preds = []
  all_targets = []
  with torch.no_grad():
    for seqs, targets, lens in val_dataloader:
      seqs = seqs.to(device)
      targets = targets.to(device)

      preds = model(seqs.type(torch.float))
      preds = torch.sigmoid(preds)

      targets = targets.type(torch.int)
      preds = preds.cpu().squeeze().numpy().T # (1, 29, L) -> (L, 29)
      targets = targets.cpu().squeeze().numpy().T

      all_preds.append(preds)
      all_targets.append(targets)


  concat_targets, concat_preds = np.concatenate(all_targets), np.concatenate(all_preds)
  concat_class_preds = (concat_preds > (thresholds)).astype(int)

  # precision, recall and F1 per label
  report = classification_report(concat_targets, concat_class_preds, 
                                 zero_division=zero_div, output_dict=True)

  with open(f"{threshold_path}/global_classification_report", "w") as f:
    json.dump(report, f)

  # confusion matrix per label
  confusion = multilabel_confusion_matrix(concat_targets, concat_class_preds)

  with open(f"{threshold_path}/confusion", "wb") as f:
    np.save(f, confusion)

  with open(f"{threshold_path}/y_probs", "wb") as f:
    np.save(f, concat_preds)

  with open(f"{threshold_path}/y_true", "wb") as f:
    np.save(f, concat_targets)


def run_crossval_training(carp_dim, w=0.1):
    if carp_dim not in ("76M", "38M", "600k"):
        raise ValueError("invalid model dimension parameter")

    CARP_DIM = "carp_{carp_dim}"
    encoding_dim = 128 if carp_dim == "600k" else 1024

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    m = CARP_CNN

    val_data = CARPDataset(data_path=BYTES_BAL_VAL_SET_FILEPATH)
    train_data = CARPDataset(data_path=BYTES_BAL_TRAIN_SET_FILEPATH)

    subset = f"CROSSVAL/BALANCED/{CARP_DIM}/weight_decay_{w}"
    train_CNN(m, criterion, "BCE", train_data, val_data, 
              carp_path=f'{CARP_IN}/{CARP_DIM}.pt', encoding_dim=encoding_dim,
              subset_dir=subset, epoch_cap=500, batch_size=16, accum_iter=4, learning_rate=5e-4,
              weight_decay=w, early_stop_patience=15)


def run_final_training(carp_dim, w=0.1):
    if carp_dim not in ("76M", "38M", "600k"):
        raise ValueError("invalid model dimension parameter")

    CARP_DIM = "carp_{carp_dim}"
    encoding_dim = 128 if carp_dim == "600k" else 1024

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    m = CARP_CNN

    test_data = CARPDataset(data_path=BYTES_BAL_TEST_SET_FILEPATH)
    train_data = CARPDataset(data_path=BYTES_BAL_TRAINVAL_SET_FILEPATH)

    subset = f"FINAL_MODDEL/BALANCED/{CARP_DIM}/weight_decay_{w}"
    train_final_CNN(m, criterion, "BCE", train_data, test_data, 
              carp_path=f'{CARP_IN}/{CARP_DIM}.pt', encoding_dim=encoding_dim,
              subset_dir=subset, epoch_cap=100, batch_size=16, accum_iter=4, learning_rate=5e-4,
              weight_decay=w)


run_crossval_training("76M")
#run_final_training("76M")