import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
import sys
@LOSSES.register_module()
class Weight_Mse_loss(nn.Module):
    def __init__(self, loss_weight=1.0,use_sigmoid=False,loss_name="Weight_Mse_loss"):
        super(Weight_Mse_loss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        # self.batch = batch
        # self.bce_loss = nn.BCELoss()

    def weight_mse_coeff2(self, target, input):     #target,input 4 1 320 320
        target = target.unsqueeze(1)
        target[target > 0] = 1
        err = ((target > 0).float() - input)
        sq_err = err ** 2
        # mean = torch.mean(sq_err)
        # return mean
        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2.0
        is_neg_err = (sign_err - 1) / -2.0

        edge_mass = torch.sum(target == 2).float()
        mid_mass = torch.sum(target == 1).float()
        empty_mass = torch.sum(target == 0).float()
        total_mass = edge_mass + empty_mass + mid_mass
        # print(edge_mass)
        # print(mid_mass)
        # print(empty_mass)
        # print(total_mass)

        weight_pos_err = 0.8  # empty_mass  / total_mass
        weight_neg_err = 0.1  # edge_mass / total_mass
        weight_mid_err = 0.2  # mid_mass / total_mass
        # print(weight_pos_err)
        # print(weight_mid_err)
        # print(weight_neg_err)
        # weight_neg_err = 0.01

        pos_part = is_pos_err * sq_err * weight_pos_err
        neg_part = is_neg_err * sq_err * weight_neg_err
        mid_part = is_pos_err * sq_err * weight_mid_err

        weighted_sq_errs = neg_part + pos_part + mid_part

        # mean = torch.mean(weighted_sq_errs)
        mean = torch.mean(weighted_sq_errs)
        # if torch.isnan(mean):
        #    mean=torch.Tensor(0)
        return mean * self.loss_weight

    def weight_mse_coeff(self, target, input):      #变动的权重
        target = target.unsqueeze(1)
        target[target > 0] = 1
        err = (target - input)
        sq_err = err ** 2

        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2
        is_neg_err = (sign_err - 1) / -2

        edge_mass = torch.sum(target)
        empty_mass = torch.sum(1 - target)
        total_mass = edge_mass + empty_mass

        weight_pos_err = empty_mass / total_mass
        weight_neg_err = edge_mass / total_mass
        # print(weight_pos_err)

        pos_part = weight_pos_err * is_pos_err * sq_err
        neg_part = weight_neg_err * is_neg_err * sq_err * 2.0

        weighted_sq_errs = neg_part + pos_part

        return torch.mean(weighted_sq_errs)

    def weight_mse_coeff3(self, target, input):
        target = target.unsqueeze(1)
        target[target>0] =1
        err = (target - input)
        sq_err = err ** 2
        # mean = torch.mean(sq_err)
        # return mean
        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2.0
        is_neg_err = (sign_err - 1) / -2.0

        # edge_mass = torch.sum(target==2).float()
        # mid_mass = torch.sum(target==1).float()
        # empty_mass = torch.sum(target==0).float()
        # total_mass = edge_mass + empty_mass + mid_mass
        # print(edge_mass)
        # print(mid_mass)
        # print(empty_mass)
        # print(total_mass)

        weight_pos_err = 0.9  # empty_mass  / total_mass 0.7
        weight_neg_err = 0.1  # edge_mass / total_mass 0.3
        # weight_mid_err = 0.2#mid_mass / total_mass
        # print(weight_pos_err)
        # print(weight_mid_err)
        # print(weight_neg_err)
        # weight_neg_err = 0.01

        pos_part = is_pos_err * sq_err * weight_pos_err
        neg_part = is_neg_err * sq_err * weight_neg_err
        # mid_part = is_neg_err * (target==1).float() * sq_err * weight_mid_err

        weighted_sq_errs = neg_part + pos_part  # + mid_part

        # mean = torch.mean(weighted_sq_errs)
        mean = torch.sum(weighted_sq_errs)
        # if torch.isnan(mean):
        #    mean=torch.Tensor(0)
        return mean

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        y_true = y_true.unsqueeze(1)
        y_true[y_true>0] =1
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def weight_bce_loss(self, target, input):
        beta = 1 - torch.mean(target)
        # alpha = 1 - torch.mean(input)
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * target

        return F.binary_cross_entropy(input, target, weights, True)

    def weight_bce_loss2(self, target, input):
        beta = 1 - torch.mean((input > 0.5).float())
        # alpha = 1 - torch.mean(input)
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * (input > 0.5).float()

        return F.binary_cross_entropy(input, target, weights, True)

    def forward(self, y_pred, y_true,weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        a = self.weight_mse_coeff3(y_true, y_pred)
        # a = self.weight_mse_coeff2(y_true, y_pred)
        # b =  self.weight_bce_loss(y_true, y_pred)
        # b = self.soft_dice_loss(y_true, y_pred)
        return a