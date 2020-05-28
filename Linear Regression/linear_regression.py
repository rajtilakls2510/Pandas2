import matplotlib.pyplot as plt
import numpy as np
class LinearRegression():
    
    def __init__(self):
        
        self.m=0
        self.b=0
    
    
    
    def fit(self,x,y):
        mean_x= x.mean()
        mean_y= y.mean()
        
        numerator=0
        denominator=0
        for i in range(len(x)):
            numerator+= (y[i]-mean_y)*(x[i]-mean_x)
            denominator+= (x[i]-mean_x)**2

        self.m=numerator/denominator
        
        self.b=mean_y-self.m*mean_x
        x1=np.linspace(x.min(),x.max(),15000)
        y1=np.array([self.predict(t) for t in x1])
        plt.scatter(x,y)
        plt.scatter(x1,y1)
        
    def predict(self,x):
        
        return self.m*x + self.b