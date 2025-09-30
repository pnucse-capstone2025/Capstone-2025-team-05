import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def R(y_true, y_pred):
    """피어슨 상관계수"""
    return pearsonr(y_true, y_pred)[0]

def SD(y_true, y_pred):
    """선형회귀 기반 표준편차"""
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true)
    lr = LinearRegression().fit(y_pred, y_true)
    y_fit = lr.predict(y_pred)
    return np.sqrt(np.sum((y_true - y_fit) ** 2) / (len(y_pred) - 1))

def RMSE_CI(y_true, y_pred, alpha=0.05, n_boot=1000, seed=42):
    """Bootstrap 기반 RMSE 95% 신뢰구간"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmse_dist = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)   
        rmse_dist.append(RMSE(np.array(y_true)[idx], np.array(y_pred)[idx]))
    lower = np.percentile(rmse_dist, 100 * (alpha / 2))
    upper = np.percentile(rmse_dist, 100 * (1 - alpha / 2))
    return lower, upper

def metrics_dict(y_true, y_pred):
    ci_lower, ci_upper = RMSE_CI(y_true, y_pred)
    return {
        'R': R(y_true, y_pred),
        'RMSE': RMSE(y_true, y_pred),
        'MAE': MAE(y_true, y_pred),
        'SD': SD(y_true, y_pred),
        '95% CI': (ci_lower, ci_upper)
    }
