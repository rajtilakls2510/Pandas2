import numpy as np


# Linear Regression Class using Multi-dimensional Gradient Descent
class LinearRegressionMGD():
    
    def __init__(self):
        
        # degree: used to store number of dimensions of the coeffcients in m 
        self.degree=0
        
        # m : similar to slope but has multiple variables. It is a vector though it is initialized to zero
        self.m=0
        
        # b : the bias( intercept )
        self.b=0
        
        # vector to keep track of the costs after every epoch
        self.costs=[]
    
    def cost(self,x,y):
        
        # calculates Mean Squared Error cost
        
        cost=0
        for i in range(len(x)):
            cost+=(self.m.dot(x[i])+self.b-y[i])**2
        return cost
    
    
    def grad(self,x,y):
        
        # calculates the gradients for the coefficients and intercepts
        
        # Initializing the gradient accumulators
        sum_m=np.array([0 for i in range(self.degree)], dtype=float)
        sum_b=0
        
        
        for i in range(len(x)):
            sum_m+= 2 * x[i] * ( self.m.dot( x[i] ) + self.b - y[i] )
            sum_b+= 2 * ( self.m.dot( x[i] ) + self.b - y[i] )
        
        return sum_m,sum_b
    
    
    def fit(self,x,y,lr=0.001, batch_size=10,epochs=200, threshold=0.00001):
        
        # training function
        # parameters: 
        # x,y : features and target
        # lr : learning rate
        # batch_size : size of the parts of the dataset to be trained
        # epochs : number of times the model has to be trained on the whole dataset
        
        
        
        # Automatically detect the number of features in the dataset
        if self.degree==0:
            self.degree=len(x[0])
            self.m=np.array([0 for i in range(self.degree)])
        
        
        
        # Initialization of gradients
        step_m=np.array([1.0 for i in range(self.degree)])
        step_b=1.0
        
        
        # If batch_size> length of dataset, make it length of dataset
        if batch_size>len(x):
            batch_size=len(x)
        
        
        # start training
        epoch=0
        while ((max(abs(step_m))>threshold) | abs((step_b)>threshold )) & (epoch<epochs):
            
            # Stop training if the epochs are complete or any of the coefficients or biases become less than the threshold 
            
            index=0
            
            # Shuffling the dataset (necessary, otherwise the gradients were oscillating)
            a=np.arange(len(x))
            np.random.shuffle(a)
            x=x[a]
            y=y[a]
            
            
            avg_step_m=np.array([1.0 for i in range(self.degree)])
            avg_step_b=1.0
            
            
            epoch+=1
            
            # training each minibatch
            for j in range(batch_size-1, len(x),batch_size):
                
                # making minibatch
                mini_batch_x = x[index:j+1,:]
                mini_batch_y = y[index:j+1]
                
                index+=batch_size
            
                # Calculating step sizes
                step_m,step_b=self.grad(mini_batch_x,mini_batch_y)

                

                # If any of the step sizes becomes infinite or NaN, don't continue
                if(np.isnan(step_m.sum()) | np.isnan(step_b)):
                    print('Nan Value Encountered! Try reducing Learning Rate by a factor of 10')
                    return

                # Applying the gradients along with the learning rate
                self.m = self.m - lr / batch_size * step_m
                self.b = self.b - lr / batch_size * step_b
                
                avg_step_m+= step_m
                avg_step_b+= step_b
                
            # Storing the cost
            self.costs.append(self.cost(x,y))
            print(f' Epoch: {epoch} Cost: {self.costs[-1]} ')
        
    
    def predict(self,x):
        
        # function to predict the incoming dataset
        
        pred=[]
        
        for i in range(len(x)):
            pred.append(self.m.dot(x[i]) + self.b)
        
        return pred