import numpy as np
import torch
import math
from torch import nn
from typing import Callable
import pdb
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve
import warnings

class MeanIoUMetric:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
        self.tp=0
        self.fp=0
        self.fn=0
        self.tn=0
        self.all_true=[]
        self.all_pred=[]

    def reset(self):
        self.all_true = []
        self.all_pred = []

    def update(self, true, pred):
        for y_true, y_pred in zip(true, pred):
            self.all_true.append(y_true)
            self.all_pred.append(y_pred)
            y_pred = (y_pred > self.threshold).astype(np.int)
            y_true = (y_true > 0).astype(np.int)
            self.tp+=np.sum((y_pred==1)&(y_true==1))
            self.fp += np.sum((y_pred == 1) & (y_true == 0))
            self.fn += np.sum((y_pred == 0) & (y_true == 1))
            self.tn += np.sum((y_pred == 0) & (y_true == 0))


    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp+self.fp>0 else 0

    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn>0 else 0

    def AP(self):
        y_true=np.concatenate([layer.flatten() for layer in self.all_true])/255.0
        y_pred=np.concatenate([layer.flatten() for layer in self.all_pred])/255.0
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = auc(recall, precision)
        return ap

    def accuracy(self):
        return (self.tp+self.tn)/(self.tp+self.fp+self.tn+self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2*precision*recall / (precision+recall) if precision+recall>0 else 0

    def miou(self):
        return self.tp/(self.tp+self.fp+self.fn) if self.tp+self.fp+self.fn>0 else 0

    def compute_ods_ois(self, y_true, y_pred_prob, thresholds):
        ods_f1 = 0
        ois_f1 = 0
        best_ods_thresh = 0

        # 对于 ODS
        for thresh in thresholds:
            total_f1 = 0
            for i in range(len(y_true)):
                tp, fp, fn = self.compute_tp_fp_fn(y_pred_prob[i]/255.0, y_true[i]/255.0, thresh)
                total_f1 += self.compute_f1_score_ods(tp, fp, fn)

            avg_f1 = total_f1 / len(y_true)
            if avg_f1 > ods_f1:
                ods_f1 = avg_f1
                best_ods_thresh = thresh

        # 对于 OIS
        for i in range(len(y_true)):
            best_f1 = 0
            for thresh in thresholds:
                tp, fp, fn = self.compute_tp_fp_fn(y_pred_prob[i]/255.0, y_true[i]/255.0, thresh)
                f1 = self.compute_f1_score_ods(tp, fp, fn)
                if f1 > best_f1:
                    best_f1 = f1
            ois_f1 += best_f1

        ois_f1 /= len(y_true)

        return best_ods_thresh, ods_f1, ois_f1

    def compute_tp_fp_fn(self, y_pred, y_true, thresh):
        y_pred = (y_pred >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp, fp, fn

    def compute_f1_score_ods(self, tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp+fn > 0 else 0
        f1=2*(precision*recall)/(precision+recall) if precision+recall>0 else 0
        return f1

    def compute(self):
        thresholds = np.linspace(0, 1, 100)
        _, ods_f1, ois_f1 = self.compute_ods_ois(self.all_true,  self.all_pred, thresholds)
        return {"precision":self.precision(), "recall":self.recall(),
                "accuracy":self.accuracy(), "f1_score": self.f1(),
                "miou": self.miou(),
                "ods_f1": ods_f1, "ois_f1": ois_f1,
                "ap": self.AP()}