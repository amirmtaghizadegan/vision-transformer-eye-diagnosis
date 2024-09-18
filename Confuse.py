import numpy as np

class ConfuseMatrix:
    def __init__(self, y, predict, eps = 1e-5):
        self.cm = np.zeros((y.shape[-1], y.shape[-1]))
        for i in range(y.shape[0]):
            self.cm[y[i].astype('bool'), predict[i].astype('bool')] += 1
        self.eps = eps

    def acc(self):
        return (self.cm * np.eye(self.cm.shape[0])).sum()/(self.cm.sum() + self.eps)

    
    def prec(self):
        return (self.cm * np.eye(self.cm.shape[0])).sum(0) / (self.cm.sum(0) + self.eps)
    

    def spec(self):
        cm = self.cm
        tn = (cm * np.eye(cm.shape[0])).sum() - (cm * np.eye(cm.shape[0])).sum(0)
        fp = cm.sum(0) - (cm * np.eye(cm.shape[0])).sum(0)
        return tn/(fp + tn + self.eps)    


    def sens(self):
        cm = self.cm
        return (cm * np.eye(cm.shape[0])).sum(1) / (cm.sum(1) + self.eps)
    

    def __add__(self, rh):
        if rh is not None:
            self.cm += rh.cm
        return self
    

    def __radd__(self, lh):
        if lh is not None:
            self.cm += lh.cm
        return self