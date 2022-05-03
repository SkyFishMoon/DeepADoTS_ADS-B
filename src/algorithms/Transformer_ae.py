import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from torch.optim import lr_scheduler
from datetime import datetime
from .algorithm_utils import Algorithm, PyTorchUtils
import math
import time

class TransformerED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'Transformer-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, output_dir: str = None, save_every_epoch: int = None, details=True, feedforward_size: tuple = (0, 0),
                 head_number: tuple = (4, 4), warmup_steps: int = 1000):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.feedforward_size = feedforward_size
        self.head_number = head_number

        self.transformer_ed = None
        self.mean, self.cov = None, None
        self.warmup = warmup_steps
        self.ckpt_dir = output_dir + '/' + 'ckpts-'+ datetime.now().strftime("%b-%d_%H-%M-%S")  + '/'
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.save_every_step = save_every_epoch

    @classmethod
    def from_opt(cls, opt, seed):
        return cls(
            opt.name,
            opt.epoch_number,
            opt.batch_size,
            opt.learning_rate,
            opt.hidden_size,
            opt.window_size,
            opt.train_gaussian_percentage,
            (opt.encoder_layer_number, opt.decoder_layer_number),
            (opt.encoder_use_bias, opt.decoder_use_bias),
            (opt.encoder_dropout, opt.decoder_dropout),
            seed,
            opt.gpu,
            opt.output_dir,
            opt.save_every_epoch,
            feedforward_size=(opt.encoder_feedforward_size, opt.decoder_feedforward_size),
            head_number=(opt.encoder_head_number, opt.decoder_head_number),
            warmup_steps=opt.warmup_steps
        )

    def fit(self, X: pd.DataFrame, tensorboard):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.transformer_ed = TransformerEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout, self.feedforward_size, self.head_number,
                                   seed=self.seed, gpu=self.gpu, window_size=self.sequence_length)
        self.to_device(self.transformer_ed)
        self.optimizer = torch.optim.Adam(self.transformer_ed.parameters(), lr=self.lr)
        def warm_decay(step):
            if step < self.warmup:
                return step / self.warmup
            return self.warmup ** 0.5 * step ** -0.5
            # return self.hidden_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, warm_decay)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

        self.transformer_ed.train()
        train_step = 0
        val_step = 0
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            train_loss = []
            for ts_batch in train_loader:
                # if train_step == 4879:
                #     a = 1
                torch.cuda.synchronize()
                start = time.time()
                output = self.transformer_ed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.transformer_ed.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                train_loss.append(loss.item())
                tensorboard.add_scalar("compute_time/", end-start, train_step)
                tensorboard.add_scalar("train_loss_batch/", loss.item(), train_step)
                # tensorboard.add_scalar("learning_rate/", self.lr, train_step)
                self.optimizer.step()
                self.scheduler.step()
                tensorboard.add_scalar("learning_rate_batch/", self.optimizer.param_groups[0]['lr'], train_step)
                train_step = train_step + 1
            epoch_loss = torch.tensor(train_loss).mean().item()
            tensorboard.add_scalar("train_loss_epoch/", epoch_loss, epoch)
            tensorboard.add_scalar("learning_rate_epoch/", self.optimizer.param_groups[0]['lr'], epoch)

            if epoch % 1 == 0:
                val_loss = []
                for ts_batch in train_gaussian_loader:
                    output = self.transformer_ed(self.to_var(ts_batch))
                    val_loss.append(nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float())).item())
                val_loss = torch.tensor(val_loss).mean().item()
                tensorboard.add_scalar("valid_loss/", val_loss, val_step)
                torch.save({'epoch': epoch, 'state_dict': self.transformer_ed.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict()
                            }, self.ckpt_dir + '-epoch' + str(epoch) + '.pth')
                val_step = val_step + 1

        self.transformer_ed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.transformer_ed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)
    def predict(self, X: pd.DataFrame, tensorboard):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.transformer_ed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            with torch.no_grad():
                output = self.transformer_ed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.cpu().numpy())
                errors.append(error.data.cpu().numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores


class TransformerEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple, feedforward_size: tuple, head_number: tuple,
                 seed: int, gpu: int, window_size: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.dim_feedforward = feedforward_size
        self.head_number = head_number

        self.input2hidden = nn.Linear(self.n_features, self.hidden_size)
        self.to_device(self.input2hidden)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.dim_feedforward[0],
                                                        batch_first=True,
                                                        dropout=self.dropout[0], nhead=self.head_number[0])

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers[0])
        self.to_device(self.encoder)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, dim_feedforward=self.dim_feedforward[1],
                                                        batch_first=True,
                                                        dropout=self.dropout[1], nhead=self.head_number[1])

        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)
        self.position_encoding = PositionalEncoding(self.hidden_size, self.dropout[0], window_size + 1)
        self.to_device(self.position_encoding)
        self.tos = self.to_var(torch.Tensor(torch.randn(1, self.n_features)).unsqueeze(0))
        self.tos = self.tos.cuda() if gpu is not None else self.tos.cpu()

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def forward(self, ts_batch, return_latent: bool = False):
        ts_batch = ts_batch.to(torch.float32)
        batch_size = ts_batch.shape[0]
        tos = self.tos.expand(batch_size, -1, -1)
        src_input = self.position_encoding(self.input2hidden(ts_batch))
        # 1. Encode the timeseries to make use of the last hidden state.
        # enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        memory_bank = self.encoder(src_input)
        if self.training:
            # z = self.lmbda(self.memorysum(memory_bank.view(batch_size, -1))).unsqueeze(1).expand(-1, self.window_size,
            #                                                                                      -1)
            tgt_input = self.position_encoding(
                self.input2hidden(torch.cat([tos, torch.flip(ts_batch, dims=[1])], dim=1)))
            tgt_output = torch.flip(self.decoder(tgt_input, memory_bank), dims=[1])
            # g = self.sigmoid(self.gate(torch.cat([tgt_output[:, 1:, :], z], dim=-1)))
            # z_hidden = g * z + (1 - g) * tgt_output[:, 1:, :]
            output = self.hidden2output(tgt_output[:, 1:, :])
        else:
            # ys = self.to_var(torch.randn(ts_batch.shape[-1], 1).type_as(ts_batch.data))
            # z = self.lmbda(self.memorysum(memory_bank.view(batch_size, -1)))
            output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
            for i in reversed(range(ts_batch.shape[1])):
                hidden = self.decoder(tgt=self.input2hidden(tos), memory=memory_bank,
                                      tgt_mask=self.to_var(
                                          self.subsequent_mask(tos.shape[1]).type_as(ts_batch.data)).squeeze(0))
                # g_i = self.sigmoid(self.gate(torch.cat([hidden[:, -1, :], z], dim=-1)))
                # z_hidden_i = g_i * z + (1 - g_i) * hidden[:, -1, :]
                output[:, i, :] = self.hidden2output(hidden[:, -1, :])
                tos = torch.cat([tos, output[:, i, :].unsqueeze(1)], dim=1)
            # output = self.hidden2output(torch.cat([tgt_output, z], dim=-1))
        # _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        # dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        # output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        # for i in reversed(range(ts_batch.shape[1])):
        #     output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])
        #
        #     if self.training:
        #         _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
        #     else:
        #         _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, memory_bank) if return_latent else output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1), :]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)
