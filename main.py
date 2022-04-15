import glob
import os

import numpy as np
import pandas as pd

from experiments import run_extremes_experiment, run_multivariate_experiment, run_multi_dim_multivariate_experiment, \
    announce_experiment, run_multivariate_polluted_experiment, run_different_window_sizes_evaluator, run_adsb_experiment
from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED, Donut
from src.datasets import KDDCup, RealPickledDataset
from src.evaluation import Evaluator
from src.evaluation import ADSBEvaluator
import configargparse
import src.opts as opts
from src.algorithms import model
import torch
import random

RUNS = 1


def _get_parser():
    parser = configargparse.ArgumentParser(
        description='main.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opts.test_opts(parser)
    # parser.add_argument('--manifold', type=str, default=None,
    #                     choices=['Euclidean', 'Hyperboloid', 'PoincareBall'])
    # parser.add_argument('--c', type=float, default=None)
    # parser.add_argument('--act', type=str, default='selu')
    # parser.add_argument('--centroid_num', type=int, required=True)
    # parser.add_argument('--weight_decay', '-weight_decay', type=float, default=0.)
    return parser

def main():
    parser = _get_parser()

    opt = parser.parse_args()
    run_experiments(opt)


def detectors(seed):
    standard_epochs = 40
    dets = [
            # AutoEncoder(num_epochs=standard_epochs, seed=seed, gpu=0),
            # DAGMM(num_epochs=standard_epochs, seed=seed, lr=1e-4, gpu=0),
            # DAGMM(num_epochs=standard_epochs, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed, gpu=0),
            # LSTMAD(num_epochs=standard_epochs, seed=seed, gpu=0),
            LSTMED(num_epochs=standard_epochs, seed=seed, gpu=0),
            # RecurrentEBM(num_epochs=standard_epochs, seed=seed, gpu=0),
            # Donut(num_epochs=standard_epochs, seed=seed, gpu=0)
    ]

    return sorted(dets, key=lambda x: x.framework)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_experiments(opt=None):
    # Set the seed manually for reproducibility.
    opt.seeds = [20]
    output_dir = opt.output_dir
    evaluators = []
    # outlier_height_steps = 10

    # for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
    #     announce_experiment('Outlier Height')
    #     ev_extr = run_extremes_experiment(
    #         detectors, seeds, RUNS, outlier_type, steps=outlier_height_steps,
    #         output_dir=os.path.join(output_dir, outlier_type, 'intensity'))
    #     evaluators.append(ev_extr)

    announce_experiment('ADS-B Datasets')
    ev_mv = run_adsb_experiment(
        detectors, opt.seeds, RUNS,
        output_dir=os.path.join(output_dir, 'multivariate'), opt=opt)
    evaluators.append(ev_mv)

    # announce_experiment('Multivariate Datasets')
    # ev_mv = run_multivariate_experiment(
    #     detectors, seeds, RUNS,
    #     output_dir=os.path.join(output_dir, 'multivariate'))
    # evaluators.append(ev_mv)
    #
    # for mv_anomaly in ['doubled', 'inversed', 'shrinked', 'delayed', 'xor', 'delayed_missing']:
    #     announce_experiment(f'Multivariate Polluted {mv_anomaly} Datasets')
    #     ev_mv = run_multivariate_polluted_experiment(
    #         detectors, seeds, RUNS, mv_anomaly,
    #         output_dir=os.path.join(output_dir, 'mv_polluted'))
    #     evaluators.append(ev_mv)
    #
    #     announce_experiment(f'High-dimensional multivariate {mv_anomaly} outliers')
    #     ev_mv_dim = run_multi_dim_multivariate_experiment(
    #         detectors, seeds, RUNS, mv_anomaly, steps=20,
    #         output_dir=os.path.join(output_dir, 'multi_dim_mv'))
    #     evaluators.append(ev_mv_dim)
    #
    # announce_experiment('Long-Term Experiments')
    # ev_different_windows = run_different_window_sizes_evaluator(different_window_detectors, seeds, RUNS)
    # evaluators.append(ev_different_windows)

    for ev in evaluators:
        ev.plot_single_heatmap()


def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    results = pd.DataFrame()
    datasets = [KDDCup(seed=1)]
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        datasets[0] = KDDCup(seed)
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)


def different_window_detectors(seed):
    standard_epochs = 40
    dets = [LSTMAD(num_epochs=standard_epochs)]
    for window_size in [13, 25, 50, 100]:
        dets.extend([LSTMED(name='LSTMED Window: ' + str(window_size), num_epochs=standard_epochs, seed=seed,
                            sequence_length=window_size),
                     AutoEncoder(name='AE Window: ' + str(window_size), num_epochs=standard_epochs, seed=seed,
                            sequence_length=window_size)])
    return dets


if __name__ == '__main__':
    main()
