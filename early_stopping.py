import torch

# adapted from https://github.com/Bjarten/early-stopping-pytorch
# accessed 07/05/2023
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, model_path='checkpoint.pt', opt_path='opt_checkpoint.pt',
                 trace_func=print, last_val_loss=np.Inf):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = last_val_loss
        self.delta = delta
        self.model_path = model_path
        self.opt_path = opt_path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, optimizer):
        score = -val_loss

        if self.patience is None:
          self.save_checkpoint(val_loss, model, optimizer)
          return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path)
        torch.save(optimizer.state_dict(), self.opt_path)
        self.val_loss_min = val_loss