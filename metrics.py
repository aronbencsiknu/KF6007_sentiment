import torch

class Metrics():
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
    
    # calculate the confusion matrix
    def perf_measure(self, y_actual, y_hat):

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                self.TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                self.FP += 1
            if y_actual[i]==y_hat[i]==0:
                self.TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
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
    def accuracy(self, output, y_true):
        ps = torch.exp(output)
        top_class = torch.argmax(ps, dim=1)
        equals = top_class == y_true

        return torch.mean(equals.type(torch.FloatTensor)).item()
    
    def acc(self, pred,label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()
        
        