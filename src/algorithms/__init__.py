from .dagmm import DAGMM
from .donut import Donut
from .autoencoder import AutoEncoder
from .lstm_ad import LSTMAD
from src.algorithms.lstm_enc_dec_axl import LSTMED
from .rnn_ebm import RecurrentEBM
from .lstm_vae import LSTMVED
from .Transformer_ae import TransformerED
from .Transformer_vae import TransformerVED

model = {"lstm_ed": LSTMED, "lstm_ved": LSTMVED, "dagmm_lstm": DAGMM, "dagmm_fc": DAGMM,
         'transformer_ed': TransformerED, "transformer_ved": TransformerVED,
         "recurrent_ebm": RecurrentEBM, "donut": Donut, "lstm_ad": LSTMAD, 'fc_ed': AutoEncoder}


__all__ = [
    'AutoEncoder',
    'DAGMM',
    'Donut',
    'LSTMAD',
    'LSTMED',
    'RecurrentEBM'
]

