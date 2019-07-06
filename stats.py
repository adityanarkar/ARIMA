import numpy as np
import collections


class stats(object):

    def __init__(self, finalData=[], actual=[], predictions=[]):
        self.finalData = finalData
        self.actual = actual
        self.predictions = predictions
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.actualCalls = []
        self.predCalls = []

    def getPredictionBuySellCall(self):
        var = self.actual >= self.predictions
        return [x * 1 for x in var]

    def getActualCall(self):
        return self.finalData[:, 1]

    def getTP(self):
        sumT = collections.Counter(self.actualCalls * self.predCalls)[1]
        return sumT

    def getFP(self):
        count = 0
        for i in range(len(self.actualCalls)):
            if (self.actualCalls[i] == 0) and (self.predCalls[i] == 1):
                count += 1
        return count

    def getTN(self):
        return collections.Counter((self.actualCalls + self.predCalls))[0]

    def getFN(self):
        count = 0
        for i in range(len(self.actualCalls)):
            if (self.actualCalls[i] == 1) and (self.predCalls[i] == 0):
                count += 1
        return count

    def getConfusionMatrix(self):
        return self.getTP(), self.getFP(), self.getTN(), self.getFN()

    def getAccuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def getPrecision(self):
        return self.tp / (self.tp + self.fp)

    def getRecall(self):
        return self.tp / (self.tp + self.fn)

    def getStats(self):
        self.actualCalls = self.getActualCall()
        self.predCalls = self.getPredictionBuySellCall()
        self.tp, self.fp, self.tn, self.fn = self.getConfusionMatrix()
        accuracy = self.getAccuracy()
        precision = self.getPrecision()
        recall = self.getRecall()
        return accuracy, precision, recall
