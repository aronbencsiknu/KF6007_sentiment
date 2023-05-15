import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from construct_dataset import idx_to_class

class Metrics():
    def __init__(self, num_classes):
        self.confusion_matrix = torch.zeros(num_classes, num_classes)
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def display_report(self):

        self.calculate_tp_fp_tn_fn()
        precision = self.precision()
        recall = self.recall()
        f1_score = self.f1_score()
        accuracy = self.accuracy()

        print("\nAccuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1_score)
    
        # display confusion matrix
        self.confusion_matrix = self.confusion_matrix.numpy()
        normalized_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Create a figure and plot the original confusion matrix
        plt.figure(figsize=(5,10))
        plt.subplot(2, 1, 1)
        sns.heatmap(self.confusion_matrix, annot=True, linewidths=.5, square=True, fmt='g', cmap='viridis')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix')

        # Plot the normalized confusion matrix
        plt.subplot(2, 1, 2)
        sns.heatmap(normalized_matrix, annot=True, linewidths=.5, square=True, fmt='.2f', cmap='viridis')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Normalized Confusion matrix')

        # Adjust the layout and display the figure
        plt.tight_layout()
        plt.show()

    
    # calculate the confusion matrix
    def increment_confusion_matrix(self, target, logps):
        top_class = self.get_predicted(logps)
        self.confusion_matrix += confusion_matrix(target, top_class)
      
    def calculate_tp_fp_tn_fn(self):
        self.FP = np.asarray(self.confusion_matrix.sum(axis=0)) - np.diag(self.confusion_matrix)
        self.FN = np.asarray(self.confusion_matrix.sum(axis=1)) - np.diag(self.confusion_matrix)
        self.TP = np.diag(self.confusion_matrix)
        self.TN = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)

        print("\nTP: ", self.TP)
        print("FP: ", self.FP)
        print("TN: ", self.TN)
        print("FN: ", self.FN)

    # calculate the precision
    def precision(self, average=True):
        
        precision = self.TP/(self.TP+self.FP)
        if average:
            return np.average(precision)
        return precision

    # calculate the recall
    def recall(self, average=True):
        recall = self.TP/(self.TP+self.FN)
        if average:
            return np.average(recall)
        return recall
    
    # calculate the harmonic mean of precision and recall
    def f1_score(self, average=True):
        f1_score = 2*(self.precision()*self.recall())/(self.precision()+self.recall())
        if average:
            return np.average(f1_score)
        return f1_score

    # calculate the accuracy of entire dataset
    def accuracy(self, average=True):
        accuracy = (self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN)
        if average:
            return np.average(accuracy)
        return accuracy
    
    # calculate the accuracy during training
    def acc(self, logps=None, labels=None):
        top_class = self.get_predicted(logps)
        equals = top_class == labels.view(*top_class.shape)
        
        return torch.mean(equals.type(torch.FloatTensor)).item()
    
    def get_predicted(self, logps):
        ps = torch.exp(logps)
        _, top_class = ps.topk(1, dim=1)
        return top_class
    

        
        