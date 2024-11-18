import ipdb
import torch.nn as nn
import torch
from ..builder import LOSSES
@LOSSES.register_module()
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True,use_sigmoid=False,loss_weight=1.0,loss_name='dice_bec_loss',):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        # Binary CrossEntropyLoss，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类
        self.bce_loss = torch.nn.BCELoss()
        self.loss_name = loss_name
        # self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self,y_pred,y_true):
        smooth = 0.0  # may change
        if self.batch:
            # input:输入一个tensor
            # dim:要求和的维度，可以是一个列表
            # 若不写，则对整张图片求和
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            # sum(-1)和sum(1)
            # 用途：求数组每一列的和，等价于 sum(axis=1)
            # sum(0)
            # 用途：求数组每一行的和，等价于sum(axis=0)
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self,  y_pred,y_true):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self,y_pred,y_true,ignore_index,weight=None):

        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.to(torch.float)
        y_true = y_true.to(torch.float)
        # a = self.bce_loss(y_pred,y_true)
        # b = self.soft_dice_loss(y_pred,y_true )
        # return a + b
        return self.bce_loss(y_pred,y_true) + self.soft_dice_loss(y_pred,y_true )