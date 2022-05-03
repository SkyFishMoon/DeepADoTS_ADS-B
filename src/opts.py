def config_opts(parser):
    parser.add('--config', '-config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('--save_config', '-save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')
    parser.add('--output_dir', '-output_dir', required=True, type=str)
    parser.add('--tensorboard_dir', '-tensorboard_dir', required=True, type=str)


def model_opts(parser):
    parser.add('--name', '-name', required=True,
               type=str, choices=['lstm_ed', 'lstm_ved', 'transformer_ed',
                                  'transformer_ved', 'dagmm_lstm', 'dagmm_fc',
                                  'recurrent_ebm', 'donut', 'lstm_ad', 'fc_ed', 'transformer_ved_local', 'lstm_ved_local'])
    parser.add('--feature_size', '-feature_size', required=True,
               type=int)
    parser.add('--hidden_size', '-hidden_size', required=True,
               type=int)
    parser.add('--encoder_feedforward_size', '-encoder_feedforward_size', default=0,
               type=int)
    parser.add('--decoder_feedforward_size', '-decoder_feedforward_size', default=0,
               type=int)
    parser.add('--latent_length', '-latent_length', default=0,
               type=int)
    parser.add('--latent_number', '-latent_number', default=0,
               type=int)
    parser.add('--encoder_layer_number', '-encoder_layer_number', required=True,
               type=int)
    parser.add('--decoder_layer_number', '-decoder_layer_number', required=True,
               type=int)
    parser.add('--encoder_head_number', '-encoder_head_number', default=0,
               type=int)
    parser.add('--decoder_head_number', '-decoder_head_number', default=0,
               type=int)
    parser.add('--encoder_dropout', '-encoder_dropout', default=0.0,
               type=float)
    parser.add('--decoder_dropout', '-decoder_dropout', default=0.0,
               type=float)
    parser.add('--encoder_use_bias', '-encoder_use_bias', action='store_true')
    parser.add('--decoder_use_bias', '-decoder_use_bias', action='store_true')


def train_opts(parser):
    parser.add('--epoch_number', '-epoch_number', required=True,
               type=int)
    parser.add('--batch_size', '-batch_size', required=True,
               type=int)
    parser.add('--window_size', '-window_size', required=True,
               type=int)
    parser.add('--train_gaussian_percentage', '-train_gaussian_percentage', required=True,
               type=float)
    parser.add('--learning_rate', '-learning_rate', type=float, default=1.0,
               help="Starting learning rate. "
                    "Recommended settings: sgd = 1, adagrad = 0.1, "
                    "adadelta = 1, adam = 0.001")
    parser.add('--learning_rate_decay', '-learning_rate_decay',
               type=float, default=0.5,
               help="If update_learning_rate, decay learning rate by "
                    "this much if steps have gone past"
                    "start_decay_steps")
    parser.add('--start_decay_steps', '-start_decay_steps',
               type=int, default=50000,
               help="Start decaying every decay_steps after"
                    "start_decay_steps")
    parser.add('--decay_steps', '-decay_steps', type=int, default=10000,
               help="Decay every decay_steps")

    parser.add('--decay_method', '-decay_method', type=str, default="none",
               choices=['noam', 'noamwd', 'rsqrt', 'none'],
               help="Use a custom decay rate.")
    parser.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
               help="Number of warmup steps for custom decay.")
    parser.add('--gpu', '-gpu', type=int, default=None)
    parser.add('--early_stopping', '-early_stopping', type=int, default=0,
              help='Number of validation steps without improving.')
    parser.add('--threshold_step', '-threshold_step', type=int, default=100)
    parser.add('--save_every_epoch', '-save_every_epoch', type=int, default=0)

def test_opts(parser):
    parser.add('--test_ckpts_number', type=int, default=10)