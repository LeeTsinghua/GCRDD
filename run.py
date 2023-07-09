import os
import urllib
import zipfile
import numpy as np
import pandas as pd
import torch
import configparser
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.dataset.metr_la import METRLADatasetLoader
from gluonts.dataset.common import ListDataset
from pts.model.graph_time_grad import GraphTimeGradEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions, Evaluator
from gluonts.evaluation import MultivariateEvaluator
from pts.model.graph_time_grad.graph_recurrent import GConvGRU
device = torch.device("cuda")

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
config = configparser.ConfigParser()
config.read('config.ini')

raw_data = config['Paths']['raw_data']

raw_data_dir = os.path.join(os.getcwd(), "./Data/METR_LA/")
dataset = METRLADatasetLoader(raw_data_dir = raw_data_dir)
#A = dataset.A
#X = dataset.X
A = np.load(os.path.join(raw_data_dir, "adj_mat.npy"))
A = (A + A.T) - np.diag(2*np.ones(207))
X = np.load(os.path.join(raw_data_dir, "node_values.npy")).transpose((1,2,0))
X = X.astype(np.float32)
X = X[:,0,:]   # Actual speed readings
X[X==0] = 1.0    # for the purpose of being able to calculate MAP
num_nodes = A.shape[0]

edge_index = np.nonzero(A)
edge_weight = np.float32(A[edge_index])
edge_index = np.stack(edge_index, axis=0)

H = int(config['Model']['prediction_length'])
M = int(config['Model']['context_length'])

date_range = pd.date_range("2012-03-01", periods=X.shape[1], freq='5min')
period = pd.Period('2012-03-01', freq='5min')
# We use the last 10 day for test 10*288 = 2880
period_test = pd.Period('2012-06-18', freq='5min')
dataset_train = [{'target': X[:,:-H], 'start': period, 'feat_static_cat': np.array([0]), 'edge_index': edge_index, 'edge_weight': edge_weight}]
dataset_test = [{'target': X, 'start': period_test, 'feat_static_cat': np.array([0]), 'edge_index': edge_index, 'edge_weight': edge_weight}]
#print(dataset_test)
seed = 10203
np.random.seed(seed)
torch.manual_seed(seed)

estimator = GraphTimeGradEstimator(
    num_nodes=num_nodes,
    prediction_length=H,
    context_length=M,
    cell_type=config['Model']['cell_type'],
    input_size=int(config['Model']['input_size']),
    freq=config['Model']['freq'],
    loss_type=config['Model']['loss_type'],
    scaling=config.getboolean('Model', 'scaling'),
    diff_steps=int(config['Model']['diff_steps']),
    beta_end=float(config['Model']['beta_end']),
    beta_schedule=config['Model']['beta_schedule'],
    trainer=Trainer(
        device=device,
        epochs=int(config['Trainer']['epochs']),
        learning_rate=float(config['Trainer']['learning_rate']),
        num_batches_per_epoch=int(config['Trainer']['num_batches_per_epoch']),
        batch_size=int(config['Trainer']['batch_size'])
    )
)

predictor = estimator.train(dataset_train, num_workers=0)  #8

forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                predictor=predictor,
                                                num_samples=config['Forecast']['num_samples'])

forecasts_pytorch = list(forecast_it) #f for f in forecast_it)
tss_pytorch = list(ts_it)

# Let us calculate our own "RMSE", "MAE" and "MAPE"
RMSE = []
MAE  = []
MAPE = []
for tts, forecast in zip(tss_pytorch, forecasts_pytorch):
    true_target = tts.values[-H:,:]
    RMSE.append(np.sqrt(((true_target - forecast.mean)**2).mean()) )
    MAE.append((np.abs(true_target - forecast.mean)).mean())
    MAPE.append((np.abs(true_target - forecast.mean) / np.abs(true_target)).mean() )

print('RMSE: {:}'.format(RMSE[0]))
print('MAE: {:}'.format(MAE[0]))
print('MAPE: {:}'.format(MAPE[0]))