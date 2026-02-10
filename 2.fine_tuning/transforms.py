import random
import numpy as np

class add:
    def __init__(self, p):
        
        self.p = p
        
    def __call__(self, X, betashift = 0.05):
        if random.random() < self.p:
            beta = np.random.random(size=(X.shape[0],1))*2*betashift-betashift
            X = X + beta
            return X
        else:
            return X

class multi:
    def __init__(self, p):
        
        self.p = p
        
    def __call__(self, X, multishift = 0.05):
        if random.random() < self.p:
            multi = np.random.random(size=(X.shape[0],1))*2*multishift-multishift + 1
            X = multi*X
            return X
        else:
            return X

class slop:
    def __init__(self, p):
        
        self.p = p
        
    def __call__(self, X, slopeshift = 0.05):
        if random.random() < self.p:
            slope = np.random.random(size=(X.shape[0],1))*2*slopeshift-slopeshift + 1
            axis = np.array(range(X.shape[1]))/float(X.shape[1]) 
            offset = slope*(axis) - axis - slope/2. + 0.5
            X = X + offset
            return X
        else:
            return X
            
            