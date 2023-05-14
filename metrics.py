import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Metrics():
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def display_report(self, target, predicted):
        self.perf_measure(target, predicted)

        precision = self.precision()
        recall = self.recall()
        f1_score = self.f1_score()
        accuracy = self.acc(target, predicted)

        confusion_matrix = confusion_matrix(target, predicted)

        # display here
    
    # calculate the confusion matrix
    def perf_measure(self, targets, predicted):
        for i in range(len(predicted)): 
            if targets[i] == predicted[i] == 1:
                self.TP += 1
            if predicted[i] == 1 and targets[i] != predicted[i]:
                self.FP += 1
            if targets[i] == predicted[i] == 0:
                self.TN += 1
            if predicted[i] == 0 and targets[i] != predicted[i]:
                self.FN += 1

        
    # calculate the precision
    def precision(self):
        return self.TP/(self.TP+self.FP+1e-8)

    # calculate the recall
    def recall(self):
        return self.TP/(self.TP+self.FN+1e-8)
    
    # calculate the harmonic mean of precision and recall
    def f1_score(self):
        return 2*(self.precision()*self.recall())/(self.precision()+self.recall()+1e-8)

    # calculate the accuracy
    def acc(self, logps=None, labels=None):
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        return torch.mean(equals.type(torch.FloatTensor)).item()
    
    

        
        