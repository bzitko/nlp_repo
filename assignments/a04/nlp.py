import sys
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from bidict import bidict
from IPython.display import display, Markdown


class Vocabulary(bidict):
    """
    A class for indexing elements of vocabulary.
    """
    def __init__(self, pad_tok=None, bgn_tok=None, end_tok=None, unk_tok=None):
        self.pad_tok = pad_tok # 0
        self.bgn_tok = bgn_tok # 1
        self.end_tok = end_tok # 2
        self.unk_tok = unk_tok # 3

        init_vocab = {}
        for tok in [pad_tok, bgn_tok, end_tok, unk_tok]:
            if tok:
                init_vocab[tok] = len(init_vocab)
        super(Vocabulary, self).__init__(init_vocab)

    @property
    def pad_idx(self):
        """returns index of pad token"""
        return self.get(self.pad_tok, None)

    def _add(self, tok):
        """adds token and returns its index"""
        if tok in self:
            return self[tok]
        else:
            self[tok] = len(self)

    def fill(self, tokens, cutoff=None):
        """
        adds multiple tokens into vocabulary with possible cutoff of 
        tokens whose frequency is less than given parameters
        """
        if cutoff:
            counter = {}
            for tok in tokens:
                counter[tok] = counter.get(tok, 0) + 1
            tokens = {tok 
                      for tok in counter 
                      if counter[tok] >= cutoff}
        else:
            tokens = set(tokens)

        for tok in sorted(tokens):
            self._add(tok)
    
    def __getitem__(self, tok):
        """
        returns index of the token or unknown token 
        if token is not in vocabulaty
        """
        if not tok in self:
            if not self.unk_tok:
                raise Exception(f"'{tok}' is not in vocabulary")
            else:
                tok = self.unk_tok
        return super(Vocabulary, self).__getitem__(tok)
    
    def vocabularize(self, tokens) -> torch.Tensor:
        """
        returns indices tensor of given tokens
        """
        return torch.tensor([self[tok] for tok in tokens])
    
    def unvocabularize(self, indices) -> list:
        """
        returns sequence of tokens according to their indices
        """
        return [self.inverse[i] for i in indices]

    def pad(self, tokens, size, left=False):
        """
        """
        if not self.pad_tok:
            raise Exception(f"'Can't pad sequence of tokens when vocabulary hasn't pad token'")

        tokens = list(tokens)
        if self.bgn_tok:
            tokens = [self.bgn_tok] + tokens
        if self.end_tok:
            tokens = tokens + [self.end_tok]

        tokens = tokens[:size]
        tokens = self.vocabularize(tokens)
        if len(tokens) < size:
            pads = torch.tensor([self[self.pad_tok]] * (size - len(tokens)))
            if left:
                tokens = torch.cat([pads, tokens])
            else:
                tokens = torch.cat([tokens, pads])

        return tokens

    def pad_many(self, tokenized_corpus, size, left=False):
        return torch.stack([self.pad(tokens, size, left)
                            for tokens in tokenized_corpus])


class Vectorizer(object):

    def __init__(self, vocab, binary=True):
        self.vocab = vocab
        self.binary = binary
        self.max_size = max_size

    # VECTORIZE
    def vectorize(self, tokens):
        array = torch.zeros(len(self.vocab))
        for i in self.vocab.vocabularize(tokens):
            if self.binary:
                array[i] = 1
            else:
                array[i] += 1
        return array
    
    def vectorize_many(self, tokenized_corpus):
        array = [self.vectorize(tokens) for tokens in tokenized_corpus]
        return torch.stack(array)

    # UNVECTORIZE
    def unvectorize(self, array):
        bow = {}
        for i, cnt in enumerate(array):
            w = self.vocab.inv[i]
            bow[w] = int(cnt)
        return bow

    def __len__(self):
        return len(self.vocab)
    

class ConvVectorizer(object):

    def __init__(self, vocab, max_size):
        self.vocab = vocab
        self.max_size = max_size

    def vectorize(self, tokens):
        tokens = list(tokens)[:self.max_size]
        
        array = torch.zeros((len(self.vocab), self.max_size))
        for j, w in enumerate(tokens):
            i = self.vocab[w]
            array[i, j] = 1
        
        return array

    def vectorize_many(self, tokenized_corpus):
        array = [self.vectorize(tokens) for tokens in tokenized_corpus]
        return torch.stack(array)

    # UNVECTORIZE
    def unvectorize(self, array):
        tokens = [self.vocab.pad_tok] * self.max_size
        for i, j in array.nonzero():
            tokens[j] = self.vocab.inv[j]
        return tokens


    def __len__(self):
        return len(self.vocab)

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
                self._epoch_callback()
        except KeyboardInterrupt:
            print("Training interrupted")

    def predict(self, xses):
        self.model.eval()

        x_tensors = [torch.as_tensor(x).to(self.device) for x in xses]

        yhat_tensor = self.model(*x_tensors)

        self.model.train()

        return yhat_tensor.detach().cpu()
    
    
    def _epoch_callback(self):
        items = [f"Epoch: {self.total_epochs} "]
        if self.train_losses and self.train_losses[-1] is not None:
            items.append(f"train loss: {self.train_losses[-1]:.5f}")
        if self.val_losses and self.val_losses[-1] is not None:
            items.append(f"val loss: {self.val_losses[-1]:.5f}")
        if self.scheduler:
            lr = self.optimizer.param_groups[0]['lr']
            items.append(f"lr: {lr}")

        txt = " ".join(items)
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
  