import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from bidict import bidict


class Vocabulary(bidict):

    #def __init__(self, pad_tok="<pad>", bgn_tok="<bos>", end_tok="<eos>", unk_tok="<unk>"):
    def __init__(self, pad_tok=None, bgn_tok=None, end_tok=None, unk_tok=None):
        self.pad_tok = pad_tok # 0
        self.bgn_tok = bgn_tok # 1
        self.end_tok = end_tok # 2
        self.unk_tok = unk_tok # 3

        init_vocab = {}
        for tok in [bgn_tok, end_tok, unk_tok]:
            if tok:
                init_vocab[tok] = len(init_vocab)
        super(Vocabulary, self).__init__(init_vocab)
        
    def _add(self, tok):
        if tok in self:
            return self[tok]
        else:
            self[tok] = len(self)

    def fill(self, tokens, cutoff=None):
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
        if not tok in self:
            if not self.unk_tok:
                raise
            else:
                tok = self.unk_tok
        return super(Vocabulary, self).__getitem__(tok)
    
    def vocabularize(self, tokens):
        return torch.tensor([self[tok] for tok in tokens])
    
    def unvocabularize(self, indices):
        return [self.inverse[i] for i in indices]

    def lpad(self, tokens, size):
        if not self.pad_tok:
            raise
        
        if len(tokens) >= size:
            return torch.tensor(self.vocabularize(tokens[:size]))
        
        num_pads = (len(tokens) - size)
        return torch.tensor([self[self.pad_tok]] * num_pads + self.vocabularize(tokens))
    
    def rpad(self, tokens, size):
        if not self.pad_tok:
            raise
        
        if len(tokens) >= size:
            return torch.tensor(self.vocabularize(tokens[:size]))
        
        num_pads = (len(tokens) - size)
        return torch.tensor(self.vocabularize(tokens) + [self[self.pad_tok]] * num_pads)



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

        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        
        return perform_train_step

    def _make_val_step(self):

        def perform_val_step(x, y):
            self.model.eval()
            yhat = self.model(x)
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
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
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
        for epoch in range(n_epochs):
            self._train_callback()
            self.total_epochs += 1

            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def train_by_loss_change(self, loss_change_treshold=1e-3):
        loss_change = loss_change_treshold + 1
        last_loss = 10

        while loss_change >= loss_change_treshold:
            self._train_callback()
            self.total_epochs += 1

            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            
            loss_change = abs(last_loss - train_loss)

            last_loss = train_loss

    def predict(self, x):
        self.model.eval()

        x_tensor = torch.as_tensor(x).float()

        yhat_tensor = self.model(x_tensor.to(self.device))

        self.model.train()

        return yhat_tensor.detach().cpu()
    
    def _train_callback(self):
        items = [f"Epoch: {self.total_epochs} "]
        if self.train_losses:
            items.append(f"train loss: {self.train_losses[-1]:.5f}")
        if self.val_losses:
            items.append(f"val loss: {self.val_losses[-1]:.5f}")
        print(" ".join(items), end="\r")    
    
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
