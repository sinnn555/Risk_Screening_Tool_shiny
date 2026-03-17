# bp_infer.py —— 推理模块（VotingClassifier + 校准模型，与 Model.py 保存的 scaler / vot_calibrated_model 一致）
import os
import numpy as np
import pandas as pd
import joblib

SCALER_PATH = "scaler.pkl"
MODEL_PATH = "vot_model.pkl"

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("未找到 scaler.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("未找到 vot_calibrated_model.pkl")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)


def _transform_like_training(x8: np.ndarray) -> np.ndarray:
    """与训练时一致：8 列特征 MinMax 缩放后送入模型"""
    x8 = np.asarray(x8, dtype=np.float64)
    assert x8.ndim == 2 and x8.shape[1] == 8, "输入必须是 (n, 8)"
    return scaler.transform(x8)


def predict_proba(x8: np.ndarray):
    """输入 (n, 8) 特征，返回校准后的正类概率（单样本返回 float，多样本返回一维数组）"""
    X = _transform_like_training(x8)
    probs = model.predict_proba(X)[:, 1]

    if probs.shape[0] == 1:
        return float(probs[0])
    return probs
    
