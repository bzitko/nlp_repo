import sys
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, Markdown

class Vocabulary(dict):

    #def __init__(self, pad_tok="<pad>", bgn_tok="<bos>", end_tok="<eos>", unk_tok="<unk>"):
    def __init__(self, pad_tok=None, bgn_tok=None, end_tok=None, unk_tok=None):

        def _add_special_tok(tok, vocab):
            if tok:
                idx = len(vocab)
                vocab[tok] = idx
            else:
                idx = None
            return tok, idx

        init_vocab = {}
        self.pad_tok, self.pad_idx = _add_special_tok(pad_tok, init_vocab)
        self.bgn_tok, self.bgn_idx = _add_special_tok(bgn_tok, init_vocab)
        self.end_tok, self.end_idx = _add_special_tok(end_tok, init_vocab)
        self.unk_tok, self.unk_idx = _add_special_tok(unk_tok, init_vocab)
        
        super(Vocabulary, self).__init__(init_vocab)
        self.inverse = {idx: tok for tok, idx in init_vocab.items()}
        self.tok_count = {tok : 0 for tok in init_vocab}
        
    def _add(self, tok):
        if tok in self:
            return self[tok]
        else:
            i = len(self)
            self[tok] = i
            self.inverse[i] = tok
            self.tok_count[tok] = self.tok_count.get(tok, 0) + 1

    def fill(self, corpus, cutoff=None):
        if cutoff:
            counter = {}
            for tokens in corpus:
                for tok in tokens:
                    counter[tok] = counter.get(tok, 0) + 1
            tokens = {tok 
                      for tok in counter 
                      if counter[tok] >= cutoff}
        else:
            for tokens in corpus:
                for tok in tokens:
                    self._add(tok)
    
    def __getitem__(self, tok):
        if not tok in self:
            if not self.unk_tok:
                raise KeyError(f"Token '{tok}' is not in vocabulary.")
            else:
                tok = self.unk_tok
        return super(Vocabulary, self).__getitem__(tok)
    

class StepByStep(object):

    def __init__(self, model, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.train_losses = []
        self.val_losses = []

        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

        self.total_epochs = 0

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step(self):

        def perform_train_step(xses, y):
            self.model.train()
            yhat = self.model(*xses)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        
        return perform_train_step

    def _make_val_step(self):

        def perform_val_step(xses, y):
            self.model.eval()
            yhat = self.model(*xses)
            loss = self.loss_fn(yhat, y)
            return loss.item()
        
        return perform_val_step

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step = self.val_step
            scheduler = self.scheduler
        else:
            data_loader = self.train_loader
            step = self.train_step
            scheduler = None

        if data_loader is None:
            return None
        
        mini_batch_losses = []
        for *x_batches, y_batch in data_loader:
            x_batches = [x_batch.to(self.device) for x_batch in x_batches]
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batches, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        
        loss = torch.tensor(mini_batch_losses).mean()
        
        if scheduler:
            scheduler.step(loss)
        return loss
    
    @staticmethod
    def set_seed(seed=96):
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)

    def train_by_n_epochs(self, n_epochs):
        try:
            for epoch in range(n_epochs):
                self.total_epochs += 1

                train_loss = self._mini_batch(validation=False)
                self.train_losses.append(train_loss)

                with torch.no_grad():
                    val_loss = self._mini_batch(validation=True)
                    self.val_losses.append(val_loss)
                self._epoch_callback()                    
        except KeyboardInterrupt:
            print("Training interrupted")

    def train_by_loss_change(self, loss_change_treshold=1e-3):
        loss_change = loss_change_treshold + 1
        last_loss = 10

        try:
            while loss_change >= loss_change_treshold:
                self.total_epochs += 1

                train_loss = self._mini_batch(validation=False)
                self.train_losses.append(train_loss)

                with torch.no_grad():
                    val_loss = self._mini_batch(validation=True)
                    self.val_losses.append(val_loss)
                
                loss_change = abs(last_loss - train_loss)

                last_loss = train_loss
                self._epoch_callback(**{"loss change": loss_change})
        except KeyboardInterrupt:
            print("Training interrupted")

    def predict(self, xses):
        self.model.eval()
        
        x_tensors = [torch.as_tensor(x).to(self.device) for x in xses]
        with torch.no_grad():
            yhat_tensor = self.model(*x_tensors)

        self.model.train()

        return yhat_tensor.detach().cpu()
    
    
    def _epoch_callback(self, **kwargs):
        items = [f"Epoch: {self.total_epochs} "]
        if self.train_losses and self.train_losses[-1] is not None:
            items.append(f"train loss: {self.train_losses[-1]:.5f}")
        if self.val_losses and self.val_losses[-1] is not None:
            items.append(f"val loss: {self.val_losses[-1]:.5f}")
        if self.scheduler:
            lr = self.optimizer.param_groups[0]['lr']
            items.append(f"lr: {lr}")

        for k, v in kwargs.items():
            items.append(f"{k}: {v}")

        txt = ", ".join(items)
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write(txt)
        #print(txt, end="\r")   
    
    def reset_parameters(self):
        with torch.no_grad():
            for layer in self.model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()


    
    def plot_losses(self, ylog=True):
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        if ylog:
            plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()


def tensor2md(tensor, round=4, latex=False):
    s = len(tensor.shape)
    
    if s == 0:
        num = tensor.item()
        if isinstance(num, int):
            return str(num)
        if isinstance(num, str):
            return num
        return f"{float(num):.{round}}"

    txt = r"\begin{bmatrix} "
    if s % 2 == 0:
        m, n = tensor.shape[:2]
        rows = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(tensor2md(tensor[i, j], round=round))
            rows.append(r" & ".join(row))
        txt += r" \\ ".join(rows)
    elif s % 2 == 1:
        n = tensor.shape[0]
        row = []
        for i in range(n):
            row.append(tensor2md(tensor[i], round=round))
        txt += r" & ".join(row)
    txt += r"\end{bmatrix}"
    if latex:
        txt = "$" + txt + "$"
    return txt

def mdprint(*args, round=4, end=" "):
    txt = []
    for arg in args:
        if isinstance(arg, (torch.Tensor, np.ndarray)):
            txt.append(tensor2md(arg, round=round, latex=True))
        else:
            txt.append(arg)
    display(Markdown(end.join(txt)))

def allclose(a, b, atol=1e-4):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    return torch.allclose(a, b, atol=atol)
  