import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    
    
    
    def __init__(self,n_estimators=50,sample_size=0.7,criterion='gini',splitter='best' ,max_depth=None):
        
        # Constructor: Parameters:
        # n_estimators: Number of Estimators(Number of Decision Tree Classifiers for the Forest)
        # sample_size: The Fraction of the dataset to be randomly drawn for training each estimator (0.1<=sample_size<=1)
        # criterion: criterion of Decision Tree Classifiers( Applied to all estimators )
        # splitter: splitter of Decision Tree Classifiers( Applied to all estimators )
        # max_depth: max_depth of Decision Tree Classifiers( Applied to all estimators )
        
        
        
        self.n_estimators=n_estimators
        
        # limit sample_size if it is out of bounds
        
        if sample_size<=0:
            self.sample_size=0.1
        elif sample_size>1:
            self.sample_size=1
        else:
            self.sample_size=sample_size
        
        # Initialize the estimators
        
        self.estimators=[DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth) for i in range(self.n_estimators)]
        
    def convert_to_numpy_array(self,arr):
        
        # function to convert any iterator to numpy array
        # parameter: arr: iterator to convert to numpy array
        # If it is already a numpy array, it will be returned
        
        if type(arr) == np.ndarray:
            return arr
        elif (type(arr)==pd.core.frame.DataFrame) | (type(arr)==pd.core.series.Series):
            return arr.values

        return np.array(arr) 
    
    def get_random_samples(self,dataset: np.ndarray):
        
        # function to return random samples from the dataset
        # parameters:
        # dataset: the dataset from where random samples are to be drawn
        
        # Returns : the indices of the rows of the dataset

        num_samples=int(len(dataset)*self.sample_size)

        rows=np.random.randint(low=0,high=len(dataset),size=num_samples)

        return rows
    
    def fit(self,X_train,y_train):
        
        # function to train the model
        # parameters:
        # X_train: the features of the data
        # y_train: the labels of the data
          
        
        # Converting dataset to numpy array
        train=self.convert_to_numpy_array(X_train)
        labels=self.convert_to_numpy_array(y_train)
        

        for i in range(self.n_estimators):
            
            # sampling random rows from the dataset
            sample_rows=self.get_random_samples(train)
            sample_train=train[sample_rows]
            sample_labels=labels[sample_rows]
            
            #Training each estimator with the random sample drawn
            self.estimators[i].fit(sample_train,sample_labels)
        
    
    def aggregate(self,l):
        
        # function to find out the most frequent element of anumpy array
        # parameters:
        # l: the numpy array whose most frequent element is to be found
        
        # Returns: the most frequent element
        
        
        unique, counts = np.unique(l, return_counts=True)
        return l[np.argmax(counts)]

    def predict(self,X_test):
        
        # function to predict the testing set labels
        # parameters: 
        # X_test: testing dataset
        
        
        # converting the testing set to numpy array
        X_test=self.convert_to_numpy_array(X_test)
        
        
        # list to store predictions
        predictions=[]
        
        # list to store final predicted labels
        labels=[]
        
        
        # Predictions by each estimators are stored in predictions list
        for i in range(self.n_estimators):
            predictions.append(self.estimators[i].predict(X_test))
            
            
        # Transposing the predictions to have the prediction of a single row in a single list 
        predictions=np.array(predictions).T
        
        # Finding the aggregated results for each row and storing them in labels
        for prediction in predictions:
            labels.append(self.aggregate(prediction))
           
        return np.array(labels)