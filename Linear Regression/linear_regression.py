import matplotlib.pyplot as plt
import numpy as np

# Linear Regression class with Ordinary Least Squares Regression 
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
    

    
# Linear Regression Class using Gradient Descent
class LinearRegressionGD():
    
    def __init__(self):
        
        self.m=0
        self.b=0
    
    def ss_prime(self,x,y):
        
        sum_m=0
        sum_b=0
        
        
        for i in range(len(x)):
            sum_m+=2*x[i]*(self.m*x[i]+self.b-y[i])
            sum_b+=2*(self.m*x[i]+self.b-y[i])
            
        
        return sum_m,sum_b
    #not(np.isnan(step_m)) & (not(np.isnan(step_b))) &
    def fit(self,x,y,lr=0.1):
        
        # Initializations
        self.m=0
        self.b=0
        step_m=1.0
        step_b=1.0
        
        
        while (abs(step_m)>0.01) | abs((step_b)>0.01 ):
            
            # Calculating step sizes
            step_m,step_b=self.ss_prime(x,y)
            
            # If any of the step sizes, becomes high, don't continue
            if(np.isnan(step_m) | np.isnan(step_b)):
                print('Nan Value Encountered! Try reducing Learning Rate by a factor of 10')
                break
            
            # Applying the step along with the learning rate
            self.m=self.m-lr*step_m
            self.b=self.b-lr*step_b
        
        
        # Plotting the training points with the generated regression line
        x1=np.linspace(x.min(),x.max(),15000)
        y1=np.array([self.predict(t) for t in x1])
        plt.scatter(x,y)
        plt.scatter(x1,y1)
        
    def predict(self,x):
        
        return self.m*x + self.b