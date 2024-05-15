#!/usr/bin/python3
# tain.py
# Xavier Vasques 13/04/2021

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import preprocessing
import statsmodels.api as sm
import warnings
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import datetime
import torch.nn as nn
from itertools import combinations
from scipy.special import expit
from typing import List
import math
import torch
import torch.nn as nn
from itertools import combinations
from scipy.special import expit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class ListNet(torch.nn.Module):
    """Создаем PyTorch модель для ListNet."""

    def __init__(self, num_input_features, hidden_dim, dropout_prob):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            torch.nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_1):
        logits = self.model(input_1)
        return logits




class ListNetTrain:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10, dropout_prob=.2):
        self._prepare_data() #подготовка данных
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs
        self.dropout_prob = dropout_prob

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim, dropout_prob)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_model(self, path='listnet_wb.pth'):
        f = open(path, 'wb')
        torch.save(self.model.state_dict, f)

    def load_model(self, path='listnet_wb.pth'):
        f = open(path, 'rb')
        state_dict = torch.load(f)
        self.model.load_state_dict(state_dict)
        
    def predict(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model(x)

    def _get_data(self, train_df=train_df, test_df=test_df) -> List[np.ndarray]:
        """
        Полученныe данные преобразуются в  np массив признаков, меток
        и вектор query_id для дальнейшего обучения и теста
        """
        train_df, test_df = train_df, test_df
        c = train_df.drop(['report_date','rn','target','query_id']).columns
        X_train, X_test = train_df[c].to_numpy(), test_df[c].to_numpy()
        y_train, y_test = train_df['target'].to_numpy(), test_df['target'].to_numpy()
        query_ids_train = train_df['query_id'].to_numpy()
        query_ids_test = test_df['query_id'].to_numpy()

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        """
        Функция подготавливает данные для обучения, превращает полученный DataSet
        в FloatTensor.
        """
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()

        X_train = self._scale_features_in_query_groups(
            X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(
            X_test, self.query_ids_test)

        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)

        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        """
        Стандартизирует данные для каждого query_id.
        """
        for cur_id in np.unique(inp_query_ids):
            mask = inp_query_ids == cur_id
            tmp_array = inp_feat_array[mask]
            scaler = StandardScaler()
            inp_feat_array[mask] = scaler.fit_transform(tmp_array)

        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int, dropout_prob: float) -> torch.nn.Module:
        torch.manual_seed(123)
        net = ListNet(num_input_features=listnet_num_input_features,
                      hidden_dim=listnet_hidden_dim, dropout_prob=dropout_prob)
        return net

    def fit(self) -> List[float]:
        metrics = []
        for epoch_no in range(1, self.n_epochs + 1):
            start = datetime.datetime.now()
            self._train_one_epoch()
            finish = datetime.datetime.now()
            ep_metric = self._eval_test_set()
            metrics.append(ep_metric)
            print(f"Epoch {epoch_no}/{self.n_epochs}")
            print('Время исполнения: ' + str(finish - start))
        return metrics
    
    
    def _train_one_epoch(self):
        """Обучение проходит для каждого запроса. Создаем маску индексов для обучения"""

        self.model.train()
        for cur_id in np.unique(self.query_ids_train):
            mask_train = self.query_ids_train == cur_id
            batch_X = self.X_train[mask_train]
            batch_ys = self.ys_train[mask_train]

            self.optimizer.zero_grad()
            batch_pred = self.model(batch_X).reshape(-1, )
            batch_loss = self._calc_loss(batch_ys, batch_pred)
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        """Высчитываем функцию потерь"""

        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))



    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcg_4 = []
            ndcg_8 = []
            ndcg_12 = []
            p_at_1 = []
            p_at_4 = []
            p_at_12 = []
            rr = []
            map_1 = []
            map_4 = []
            map_12 = []
            for cur_id in np.unique(self.query_ids_test):
                mask = self.query_ids_test == cur_id
                X_test_tmp = self.X_test[mask]
                valid_pred = self.model(X_test_tmp)
                test_y = self.ys_test[mask]
                ndcg_score_4 = self._ndcg_k(
                    test_y, valid_pred, 4)
                ndcg_score_8 = self._ndcg_k(
                    test_y, valid_pred, 8)
                ndcg_score_12 = self._ndcg_k(
                    test_y, valid_pred, 12)
                if np.isnan(ndcg_score_4):
                    ndcg_4.append(0)
                    continue
                ndcg_4.append(ndcg_score_4)
                if np.isnan(ndcg_score_8):
                    ndcg_8.append(0)
                    continue
                ndcg_8.append(ndcg_score_8)
                if np.isnan(ndcg_score_12):
                    ndcg_12.append(0)
                    continue
                ndcg_12.append(ndcg_score_12)
                p_at_1.append(self._precission_at_k(
                    test_y, valid_pred))
                p_at_4.append(self._precission_at_k(
                    test_y, valid_pred, k=4))
                p_at_12.append(self._precission_at_k(
                    test_y, valid_pred, k=12))
                rr.append(self._reciprocal_rank(
                    test_y, valid_pred))
                map_1.append(self._average_precision(
                    test_y, valid_pred,1))
                map_4.append(self._average_precision(
                    test_y, valid_pred,4))
                map_12.append(self._average_precision(
                    test_y, valid_pred,12))
            return {
                    'ndcg_4':np.mean(ndcg_4),
                    'ndcg_8':np.mean(ndcg_8),
                    'ndcg_12':np.mean(ndcg_12),
                    'p_at_1':np.mean(p_at_1),
                    'p_at_4':np.mean(p_at_4),
                    'p_at_12':np.mean(p_at_12),
                    'rr':np.mean(rr),
                    'map_at_1':np.mean(map_1),
                    'map_at_4':np.mean(map_4),
                    'map_at_12':np.mean(map_12),
                   }


    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        def dcg(ys_true, ys_pred):
            _, argsort = torch.sort(ys_pred, descending=True, dim=0)
            argsort = argsort[:ndcg_top_k]
            ys_true_sorted = ys_true[argsort]
            ret = 0
            for i, l in enumerate(ys_true_sorted, 1):
                ret += (2 ** l - 1) / math.log2(1 + i)
            return ret
        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)
        return (pred_dcg / ideal_dcg).item()

    def _precission_at_k(self, ys_true, ys_pred, k=1) -> float:
        if ys_true.sum() == 0:
            return 0
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        hits = ys_true_sorted[:k].sum()
        prec = hits / min(ys_true.sum(), k)
        return float(prec)


    def _reciprocal_rank(self, ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]

        for idx, cur_y in enumerate(ys_true_sorted, 1):
            if cur_y == 1:
                return 1 / idx
        return 0


    def _average_precision(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, k=1) -> float:
        if ys_true.sum() == 0:
            return 0
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        rolling_sum = 0
        num_correct_ans = 0

        for idx, cur_y in enumerate(ys_true_sorted[:k], start=1):
            if cur_y == 1:
                num_correct_ans += 1
                rolling_sum += num_correct_ans / idx
        if num_correct_ans == 0:
            return 0
        else:
            return rolling_sum / num_correct_ans
        
    def feature_importance(self):
        self.model.eval()
        feature_importance = {}
        with torch.no_grad():
            # Iterate through each layer in the model
            for layer in self.model.model:
                # Check if the layer is a linear layer
                if isinstance(layer, torch.nn.Linear):
                    # Retrieve the weights of the linear layer
                    weights = layer.weight.data.numpy().flatten()  # Flatten the weights
                    norm_weights = abs(weights) / abs(weights).sum()  # Normalize the weights across features
                    for idx, feat in enumerate(df.drop(['report_date','rn','target','query_id']).columns):
                        # Calculate the importance as the normalized weight for each feature
                        if feat in feature_importance:
                            feature_importance[feat] += norm_weights[idx]
                        else:
                            feature_importance[feat] = norm_weights[idx]
        return feature_importance

def find_nearest_value(arr, value):
    """Находим ближайшее значение для элемента из array в value."""
    idx = np.abs(arr - value).argmin()
    return arr[idx]

def replace_with_nearest(series1, series2):
    """меняем значения на ближайшее"""
    replaced_values = [find_nearest_value(series2.values, val) for val in series1.values]
    return pd.Series(replaced_values, index=series1.index)

def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE = os.environ["MODEL_FILE"]
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
      
    # Load, read and normalize training data
    seed=123
    training = "./train_features.parquet"
    df = pl.read_parquet(training)
    imputerB = IterativeImputer(max_iter=30, random_state=seed)
    imputed_dataB = imputerB.fit_transform(df[['feature_5','feature_8','feature_10','feature_7','feature_9','feature_4','feature_1']])
    myImputer_dataB = pd.DataFrame(imputed_dataB,columns = ['feature_5','feature_8','feature_10','feature_7','feature_9','feature_4','feature_1'])
    result_series = replace_with_nearest(myImputer_dataB['feature_8'], df['feature_8'].drop_nulls().unique().to_pandas())
    df = df.with_columns(pl.Series('feature_8imp',result_series))
    df = df.with_columns(pl.Series('feature_5imp',myImputer_dataB['feature_5']))
    df = df.drop(['feature_5','feature_8'])
    q_train, q_test =  train_test_split(df['query_id'].unique(), test_size=0.3, random_state=seed)
    train_df = df.filter(pl.col('query_id').is_in(q_train))
    test_df = df.filter(pl.col('query_id').is_in(q_test))
    feature_column = df.drop(['report_date','rn','target','query_id']).columns
    X_train, X_test = train_df[feature_column].to_numpy(), test_df[feature_column].to_numpy()
    y_train, y_test = train_df['target'].to_numpy(), test_df['target'].to_numpy()
    queries_train = train_df['query_id'].to_numpy()
    queries_test = test_df['query_id'].to_numpy()

    print("Shape of the training data")
    print('X_train - ', X_train.shape)
    print('y_train - ', y_train.shape)
     
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    listnet = ListNetTrain(n_epochs=20, listnet_hidden_dim=22, lr=0.0005, dropout_prob=.1)
    listnet.fit()
    listnet_metrics = listnet._eval_test_set()
    print(listnet_metrics)
    listnet.save_model(ath=MODEL_PATH)
    # Save model
    from joblib import dump
    dump(listnet, MODEL_PATH)
        
 
if __name__ == '__main__':
    train()
