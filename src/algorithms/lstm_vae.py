import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from torch.optim import lr_scheduler
import time
from .algorithm_utils import Algorithm, PyTorchUtils
import os


class LSTMVED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-VED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, output_dir: str = None, save_every_epoch: int = None, details=True, latent_length=20):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.latent_length = latent_length
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmved = None
        self.mean, self.cov = None, None
        self.ckpt_dir = output_dir + '/' + 'ckpts' + '/'
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
            latent_length=opt.latent_length
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

        self.lstmved = LSTMVEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu, latent_length=self.latent_length)
        self.to_device(self.lstmved)
        self.optimizer = torch.optim.Adam(self.lstmved.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

        self.lstmved.train()
        train_step = 0
        val_step = 0
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            train_loss = []
            for ts_batch in train_loader:
                torch.cuda.synchronize()
                start = time.time()
                output = self.lstmved(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                kl_loss = -0.5 * torch.mean(1 + self.lstmved.lmbda.latent_logvar -
                            self.lstmved.lmbda.latent_mean.pow(2) -
                                self.lstmved.lmbda.latent_logvar.exp())
                loss = loss + kl_loss
                # tensorboard.add_scalar("learning_rate/", self.lr, train_step)
                self.lstmved.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                train_loss.append(loss.item())
                tensorboard.add_scalar("compute_time/", end - start, train_step)
                # train_loss.append(loss)
                tensorboard.add_scalar("train_loss_batch/", loss.item(), train_step)
                tensorboard.add_scalar("train_kl_loss_batch/", kl_loss.item(), train_step)
                tensorboard.add_scalar("learning_rate_batch/", self.optimizer.param_groups[0]['lr'], train_step)
                self.optimizer.step()
                train_step = train_step + 1
            epoch_loss = torch.tensor(train_loss).mean().item()
            tensorboard.add_scalar("train_loss_epoch/", epoch_loss, epoch)
            self.scheduler.step()
            tensorboard.add_scalar("learning_rate_epoch/", self.optimizer.param_groups[0]['lr'], epoch)
            if epoch % 1 == 0:
                val_loss = []
                for ts_batch in train_gaussian_loader:
                    output = self.lstmved(self.to_var(ts_batch))
                    val_loss.append(nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float())).item())
                val_loss = torch.tensor(val_loss).mean().item()
                tensorboard.add_scalar("valid_loss/", val_loss, val_step)
                torch.save({'epoch': epoch, 'state_dict': self.lstmved.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict()
                            }, self.ckpt_dir + 'epoch' + str(epoch) + '.pth')
                val_step = val_step + 1

        self.lstmved.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmved(self.to_var(ts_batch))
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

        self.lstmved.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.lstmved(self.to_var(ts))
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


class LSTMVEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int, latent_length: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)
        self.lmbda = Lambda(self.hidden_size, self.latent_length, self.seed, self.gpu)
        self.to_device(self.lmbda)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model
        h, c = enc_hidden
        # 2. Use hidden state as initialization for our Decoder-LSTM
        h = self.lmbda(h.squeeze(0)).unsqueeze(0)
        dec_hidden = (h, c)
        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):

            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output


class Lambda(nn.Module, PyTorchUtils):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length, seed, gpu):
        super(Lambda, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean