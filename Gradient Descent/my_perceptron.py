import numpy as np


# Perceptron Class using Multi-dimensional Gradient Descent and BinaryCrossEntropy Loss
class myPerceptron():
    
    def __init__(self):
        
        # degree: used to store number of dimensions of the coeffcients in w 
        self.degree=0
        
        # w : weights (For now it is initialized to zero. Later it will be initialized to a list with length as the number of features in the training data)
        self.w=0
        
        # b : bias
        self.b=0
        
        # vector to keep track of the costs after every epoch
        self.cost_history=[]
    
    def cost(self,x,y):
        
        # calculates BinaryCrossEntropy cost
        
        cost=0
        
        for i in range(len(x)):
            y_cap=self.sigmoid(self.w.dot(x[i])+self.b)
            cost+=(y[i]*np.log(y_cap)+(1-y[i])*np.log(1-y_cap))
            
        return -(cost/len(x))
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def grad(self,x,y):
        
        # calculates the gradients for the weights and biases
        
        # Initializing the gradient accumulators
        sum_w=np.array([0 for i in range(self.degree)], dtype=float)
        sum_b=0
        
        
        for i in range(len(x)):
            y_cap=self.sigmoid(self.w.dot(x[i])+self.b)
            sum_w+=  x[i] * ( y_cap - y[i] )
            sum_b+=  ( y_cap - y[i] )
        
        return sum_w,sum_b
    
    
    def fit(self,x,y,lr=0.001, batch_size=10,epochs=200, threshold=0.00001, show_epochs=1):
        
        # training function
        # parameters: 
        # x: Features of the data. It should be a 2-D numpy array
        # y: Labels of the data. Should be a 1-D numpy array
        # lr : learning rate
        # batch_size : size of the parts of the dataset to be trained
        # epochs : number of times the model has to be trained on the whole dataset
        # threshold: Minimum gradients allowed. The Training will stop if any of the gradients go below the threshold.
        # show_epochs: The Epochs after which the training cost will be shown. 
        
        
        
        # Automatically detect the number of features in the dataset
        if self.degree==0:
            self.degree=len(x[0])
            
            # Randomly initializing the weights(Uniform Distribution) and biases
            self.w=np.random.uniform(size=self.degree)
            self.b=np.random.random()
        
        
        
        # Initialization of gradients
        step_w=np.array([1.0 for i in range(self.degree)])
        step_b=1.0
        
        
        # If batch_size> length of dataset, make it length of dataset
        if batch_size>len(x):
            batch_size=len(x)
        
        
        # start training
        epoch=0
        while ((max(abs(step_w))>threshold) | abs((step_b)>threshold )) & (epoch<epochs):
            
            # Stop training if the epochs are complete or any of the gradients become less than the threshold 
            
            index=0
            
            # Shuffling the dataset (necessary, otherwise the gradients were oscillating)
            a=np.arange(len(x))
            np.random.shuffle(a)
            x=x[a]
            y=y[a]
            
            # Stores the total Gradients for all the minibatches
            tot_step_w=np.array([0.0 for i in range(self.degree)])
            tot_step_b=0.0
            
            
            epoch+=1
            
            # training each minibatch
            for j in range(batch_size-1, len(x),batch_size):
                
                # making minibatch
                mini_batch_x = x[index:j+1,:]
                mini_batch_y = y[index:j+1]
                
                index+=batch_size
            
                # Calculating step sizes
                step_w,step_b=self.grad(mini_batch_x,mini_batch_y)

                

                # If any of the step sizes becomes infinite or NaN, don't continue
                if(np.isnan(step_w.sum()) | np.isnan(step_b)):
                    print('Nan Value Encountered! Try reducing Learning Rate by a factor of 10')
                    return

                # Applying the gradients along with the learning rate
                self.w = self.w - lr / batch_size * step_w
                self.b = self.b - lr / batch_size * step_b
                
                # Storing the Gradient for this minibatch
                tot_step_w+= step_w
                tot_step_b+= step_b
                
            # Storing the cost
            self.cost_history.append(self.cost(x,y))
            
            # Showing those epochs which are specified to be printed
            if epoch%show_epochs==0:
                print(f' Epoch: {epoch} Cost: {self.cost_history[-1]} ')
        
    
    def predict(self,x):
        
        # function to predict the incoming dataset
        
        pred=[]
        
        for i in range(len(x)):
            pred.append(1 if self.sigmoid(self.w.dot(x[i]) + self.b)>=0.5 else 0)
        
        return np.array(pred)
    