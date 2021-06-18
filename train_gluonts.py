from __future__ import print_function

import argparse
import os
import logging
from sklearn.preprocessing import MinMaxScaler
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import json
import time

from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster, Seq2SeqEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.wavenet import WaveNetEstimator

from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.trainer import Trainer
from gluonts.mx.block.encoder import Seq2SeqEncoder

from gluonts.model.predictor import Predictor

from gluonts.model.naive_2 import Naive2Predictor
from gluonts.model.npts import NPTSPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

logging.basicConfig(level=logging.DEBUG)

def parse_data_scaling(dataset):# 把所有的销售量进行归一化
    data = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for t in dataset:
        tar=np.array(t['target'])
        tar[1] = scaler.fit_transform(tar[1,:].reshape(-1,1))[:,0]
        tar[2] = scaler.fit_transform(tar[2,:].reshape(-1,1))[:,0]
        tar[13:] = scaler.fit_transform(tar[13:,:])
        datai = {FieldName.TARGET: tar[:,:-12], FieldName.START: t['start']}# 这里放弃最后12个点，即实际训练范围到7.31日下午3点
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        data.append(datai)
    return data
def parse_data_scaling2(dataset):# 把除了买一价卖一价之外的进行归一化
    data = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for t in dataset:
        tar=np.array(t['target'])
        tar[0] = scaler.fit_transform(tar[0,:].reshape(-1,1))[:,0]
        tar[1] = scaler.fit_transform(tar[1,:].reshape(-1,1))[:,0]
        tar[2] = scaler.fit_transform(tar[2,:].reshape(-1,1))[:,0]
        tar[4:8] = scaler.fit_transform(tar[4:8,:])
        tar[9:13] = scaler.fit_transform(tar[9:13,:])
        tar[13:] = scaler.fit_transform(tar[13:,:])
        datai = {FieldName.TARGET: tar[:,:-12], FieldName.START: t['start']}# 这里放弃最后12个点，即实际训练范围到7.31日下午3点
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        data.append(datai)
    return data
def parse_data_ar(dataset):
    data = []
    for t in dataset:
        datai = {FieldName.TARGET: [t['target'][3][:-12],t['target'][8][:-12]], FieldName.START: t['start']}
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        data.append(datai)
    return data
def load_json(filename):
    data = []
    with open(filename, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            datai = json.loads(line)
            data.append(datai)
    return data
def parse_data(dataset):
    data = []
    for t in dataset:
        datai = {FieldName.TARGET: t['target'], FieldName.START: t['start']}
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        data.append(datai)
    return data
def train(args):
    train = load_json('./glts_train_multi_tar.json')
    test = load_json('./glts_test_multi_tar.json')

    freq = args.freq
    prediction_length = args.prediction_length
    context_length = args.context_length
    num_timeseries = len(train)
    print('num_timeseries:', num_timeseries)
    if args.prepro=="scaling2":
        model_dir = os.path.join('/Users/lvhaoran/AWScode/gluonts/models',
                                 args.algo_name + '_lr' + str(args.learning_rate) + '_epoch' + str(
                                     args.epochs) + 'scaling_all_exp2')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train_ds = ListDataset(parse_data_scaling2(train), freq=freq, one_dim_target=False)
        test_ds = ListDataset(parse_data_scaling2(test), freq=freq, one_dim_target=False)
        if args.algo_name == 'DeepVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch)
            estimator = DeepVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=23
            )
        elif args.algo_name == 'GPVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch,hybridize=True)
            estimator = GPVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=23
            )
    elif args.prepro=="scaling":
        model_dir = os.path.join('/Users/lvhaoran/AWScode/gluonts/models',
                                 args.algo_name + '_lr' + str(args.learning_rate) + '_epoch' + str(
                                     args.epochs) + 'scaling_all_num')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train_ds = ListDataset(parse_data_scaling(train), freq=freq, one_dim_target=False)
        test_ds = ListDataset(parse_data_scaling(test), freq=freq, one_dim_target=False)
         # hybridize=False
        if args.algo_name == 'DeepVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch)
            estimator = DeepVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=23
            )
        elif args.algo_name == 'GPVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch,hybridize=True)
            estimator = GPVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=23
            )
    elif args.prepro == "ar":
        model_dir = os.path.join('/Users/lvhaoran/AWScode/gluonts/models',
                                 args.algo_name + '_lr' + str(args.learning_rate) + '_epoch' + str(
                                     args.epochs) + 'scaling_all_ar')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train_ds = ListDataset(parse_data_ar(train), freq=freq, one_dim_target=False)
        test_ds = ListDataset(parse_data_ar(test), freq=freq, one_dim_target=False)# hybridize=False
        if args.algo_name=='DeepVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              learning_rate_decay_factor=0.5, batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch)
            estimator = DeepVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=2
            )
        elif args.algo_name=='GPVAR':
            trainer = Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate,
                              learning_rate_decay_factor=0.5, batch_size=args.batch_size,
                              num_batches_per_epoch=args.num_batches_per_epoch, hybridize=True)
            estimator = GPVAREstimator(  # use multi
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                target_dim=2
            )

    predictor = estimator.train(train_ds, test_ds)
    predictor.serialize(Path(model_dir))
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo-name', type=str, default='DeepVAR')
    parser.add_argument('--prepro', type=str, default='scaling')
    parser.add_argument('--model-name', type=str, default='DeepVAR')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/training')  # os.environ['SM_CHANNEL_TRAINING']
    
    parser.add_argument('--freq', type=str, default='30T')
    parser.add_argument('--prediction-length', type=int, default=1)
    parser.add_argument('--context-length', type=int, default=48*30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-batches-per-epoch', type=int, default=1000)
    
    parser.add_argument('--use-feat-dynamic-real', action='store_true', default=False)
    parser.add_argument('--use-feat-static-cat', action='store_true', default=False)
    parser.add_argument('--cardinality', type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train(args)