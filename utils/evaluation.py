# Calculation of Classification Performance Metrics
import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        # matrix shape(num_class, num_class) with elements 0 in our match. it will be 4*4

    def Kappa(self):
        xsum = np.sum(self.confusion_matrix, axis=1)  # sum by row
        ysum = np.sum(self.confusion_matrix, axis=0)  # sum by column

        Pe = np.sum(ysum * xsum) * 1.0 / (self.confusion_matrix.sum() ** 2)
        P0 = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # predict right / all the data
        cohens_coefficient = (P0 - Pe) / (1 - Pe)

        return cohens_coefficient

    def ProducerA(self):    # PA
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def UserA(self):    # UA
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def Overall_Accuracy(self):      # OA
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # each pred right class is in diag. sum by row is the count of corresponding class
        mAcc = np.nanmean(Acc)  # Acc mean
        return mAcc, Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, IoU

    def _generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)  # valid in mask show True, ignored in mask show False
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
