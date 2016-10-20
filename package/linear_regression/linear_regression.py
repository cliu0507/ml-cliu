import numpy as np
import scipy
import os
import re
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

#Function loading training data
#Possible arguments case1: x_file, y_file (row count of x_file == row count of y_file)
#Possible arguments case2: x_file, y_file=None, y_position = -1. It means y is also inside x, y is the last element of each row
def load_training_dataset(x_file, y_file=None, y_position=None, delimiter=","):
    #Check if having y_file as argument
    #Initialize x,y matrix
    x = None
    y = None
    # Initial Num of Feature and Num of samples/examples
    num_feature = 0
    num_example = 0
    if y_file == None and y_position == -1:
        # Check whether inputfile exists
        if not os.path.isfile(x_file):
            raise Exception("Data example file %s is missing" % x_file)

        # Initial X and Y(feature matrix and y matrix)
        with open(x_file, 'r') as f:
            for i, row in enumerate(f):
                #If this row has non-digit character, skip this line
                if re.search('[a-zA-Z]', row):
                    continue
                if x is not None:
                    #Leave last element of each row as y value
                    temparr = np.fromstring(row, dtype=float, sep=delimiter)
                    line = temparr[:-1]
                    y = np.append(y, temparr[-1])
                    if num_feature == line.shape[0]:
                        x = np.vstack((x, line))
                    else:
                        raise Exception("Input samples have different feature dimensions")
                else:
                    temparr = np.fromstring(row, dtype=float, sep=delimiter)
                    x = temparr[:-1]
                    y = np.array(temparr[-1])
                    num_feature = (x.shape)[0]
    elif y_file!=None and y_position == None:
        if not os.path.isfile(x_file):
            raise Exception("Data example file %s is missing" % x_file)

        # Initial X (samples matrix)
        with open(x_file, 'r') as f:
            for i, row in enumerate(f):
                #If this row has non-digit character, skip this line
                if re.search('[a-zA-Z]', row):
                    continue
                if x is not None:
                    line = np.fromstring(row, dtype=float, sep=',')
                    if num_feature == line.shape[0]:
                        x = np.vstack((x, line))
                    else:
                        raise Exception("Input samples have different feature dimensions")
                else:
                    x = np.fromstring(row, dtype=float, sep=',')
                    num_feature = (x.shape)[0]

        # Read y from data directory
        with open(y_file, 'r') as f:
            for row in f:
                #If this row has non-digit character, skip this line
                if re.search('[a-zA-Z]', row):
                    continue
                # Catch the exception if there are non numeric character in y file
                try:
                    float(row.rstrip())
                except ValueError:
                    print "y file has non-numeric character"

                # Catch alphabet character
                if re.search('[a-z][A-Z]]', row.rstrip()):
                    raise Exception("y file has alphabet")

                if y is not None:
                    y = np.append(y, float(row.rstrip()))
                else:
                    y = np.array([float(row.rstrip())])

        # Add dummy ones to matrix x for x0
        x = np.insert(x, 0, 1, axis=1)
        # Reshape x to normal shape and theta matrix
        x = x.transpose()
    else:
        raise Exception("Wrong Argument while calling load_training_data function")
    # Check if x or training dataset is empty
    if x is not None:
        num_example = x.shape[0]
    else:
        raise Exception("Training dataset file is empty")
    # Add dummy ones to matrix x for x0
    x = np.insert(x, 0, 1, axis=1)
    # Reshape x to normal shape and theta matrix
    x = x.transpose()

    return x,y,num_example,num_feature



#Feature Scaling Function: make xi approximately in the range of [-1,1]
# x is (num_feature+1, num_example)
def feature_scaling_normalization(x , type , xMeans=None , xMin=None, xMax=None):
    #Feature Scaling for training data
    if type == "training":
        print "Feature Scaling and Normalization for %s Data:" % type
        #axis should be by dimension 1, that is, aggregate by second dimension(column), view from second dimension
        xMeans= np.mean(x, axis=1)
        xMax=np.max(x,axis=1)
        xMin=np.min(x,axis=1)
        num_feature=len(xMeans)
        #Exclude x0 (No. zero feature), since we set x0 = 1 for all examples
        for i in range(1,num_feature):
            x[i][:] = (x[i][:] - xMeans[i])/[xMax[i]-xMin[i]]
        return x,xMeans,xMax,xMin
    #Feature Scaling before predicting
    elif type == "predict":
        print "Feature Scaling and Normalization for Predict Data"
        #Check Dimension, xMeans,xMin, xMax should have dimension of (num_feature,)
        if len(xMeans) != x.shape[1]+1:
            raise Exception("Dimension Dismatch")
        #Ignore first elements, since constant x0 = 1
        xMeans = xMeans[1:]
        xMin = xMin[1:]
        xMax = xMax[1:]
        #Numpy array broadcasting
        x=(x-xMeans)/(xMax - xMin)
        #Insert Dummy x0 = 1
        x_scaled = np.insert(x,0,1,axis=1)
        return x_scaled
    else:
        raise Exception("Wrong Input Arguments for load_training_dataset function")


# x is ( num_feature+1 , num_example )
# y is ( num_example , 1 )
# theta is ( num_feature+1 , 1)
# Don't have regularized term
# alpha is learning rate
def gradientDescent(x, y, theta, m, alpha , numIterations):
    errorarr=np.zeros((numIterations,))
    for i in range(0,numIterations):
        # hypothesis is (num_example, 1)
        thetaTrans = theta.transpose()
        hypothesis = np.dot(thetaTrans,x)
        hypothesis = hypothesis.reshape(m,1)
        #Initialize updated theta matrix
        theta_updated=np.zeros(theta.shape)
        #Iterate feature from 0 to num_feature
        for j in range(0,theta.shape[0]):
            #Iterate and calculate sum of error
            derivative = 0
            error = 0
            for k in range(0,m):
                derivative += (hypothesis[k][0] - y[k][0]) * x[j][k]
                error += math.sqrt(math.pow((hypothesis[k][0] - y[k][0]),2))
            theta_updated[j][0] = theta[j][0] - alpha/m * derivative
        theta = theta_updated
        errorarr[i]=error
        print error
    #Return theta, errorarr in different num_iteration and final minimum error(last element in errorarr)
    return theta,errorarr,errorarr[-1]

#Hypothesis function has regularized term
# x is ( num_feature+1 , num_example )
# y is ( num_example , 1 )
# theta is ( num_feature+1 , 1)
# lamba_value is parameters for regularized term
# alpha is learning rate
def gradientDescent2(x, y, theta, m, alpha , numIterations, lambda_value):
    errorarr=np.zeros((numIterations,))
    for i in range(0,numIterations):
        # hypothesis is (num_example, 1)
        thetaTrans = theta.transpose()
        hypothesis = np.dot(thetaTrans,x)
        hypothesis = hypothesis.reshape(m,1)
        #Initialize updated theta matrix
        theta_updated=np.zeros(theta.shape)
        #Iterate feature from 0 to num_feature
        for j in range(0,theta.shape[0]):
            #Iterate and calculate sum of error
            error = 0
            if j == 0:
                derivative = 0
                for k in range(0,m):
                    derivative += (hypothesis[k][0] - y[k][0]) * x[j][k]
                    error += math.sqrt(math.pow((hypothesis[k][0] - y[k][0]), 2))
                theta_updated[0][0] = theta[j][0] - alpha/m * derivative
            else:
                derivative = 0
                for k in range(0,m):
                    derivative += (hypothesis[k][0] - y[k][0]) * x[j][k]
                    error += math.sqrt(math.pow((hypothesis[k][0] - y[k][0]),2))
                theta_updated[j][0] = theta[j][0] * (1- lambda_value * alpha/m ) - alpha/m * derivative
        theta = theta_updated
        errorarr[i]=error
        print error
    #Return theta, errorarr in different num_iteration and final minimum error(last element in errorarr)
    return theta,errorarr,errorarr[-1]

#This is the example implementation from stackflow, note that x, y are transpose of ours
#http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
# x is (num_example, num_feature+1)
# y is (num_example,)
# theta is (num_feature+1, )
# hypothesis is (num_example,)
# loss is (num_example,)
def gradientDescent_stackflow(x, y, theta, m, alpha, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

def chart_iteration_cost( if_regularized, x , y , theta,m , alpha , numIterations ):
    if if_regularized == "true":
        #Regularized Linear Regression
        theta, errorarr, min_error = gradientDescent2(x, y, theta, m, alpha, numIterations, 10)
    else:
        theta, errorarr, min_error = gradientDescent(x , y , theta , m , alpha , numIterations)
    plt.plot(range(1,numIterations+1),errorarr)
    plt.ylabel("min(J(theta))")
    plt.xlabel("number of iteration")
    plt.show()
    return theta, errorarr, min_error

def predict(theta, x_scaled):
    y_predict=np.dot(theta.transpose(),x_scaled.transpose())
    return y_predict





#Current file absolute path
cur_file=os.path.abspath(__file__)

#Directory moving pointers
moving_dir=os.path.dirname(cur_file)

#Find /data directory
while(not os.path.isdir(moving_dir + '/data')):
    moving_dir = os.path.dirname(moving_dir)
data_dir=moving_dir + '/data'

#Example files input filename and full path
x_file=data_dir+'/winequality-red.csv'
#y is at the end of each line in x_file or y can be in a separate file
#y_file=data_dir+'/y.csv'

#Load Training Dataset files
x,y,num_example,num_feature = load_training_dataset(x_file, None, -1, delimiter=";")

#x is a matrix with dimension (num_feature+1, num_example)
#y is a vector with length num_example , y.shape() = (num_example,)
#theta is a matrix with dimension (num_feature+1,1) theta.shape() = (num_feature+1, 1)

#Convert y to (num_example,1) matrix
y= y.reshape(num_example,1)

#Random Choose theta at first round
#theta.shape = (num_feature+1,1))
theta = np.random.rand(num_feature+1,1)

#Feature Scaling and Normalization
x,xMeans,xMax,xMin = feature_scaling_normalization(x , "training")

#Call gradient descent function to Minimize J(theta)
#(theta,_,_)=gradientDescent(x, y , theta, num_example, 0.01 , 50000)
theta, errorarr, min_error = chart_iteration_cost( "false", x, y , theta, num_example , 0.5 , 500)

#Call Predict function
#[7.4,0.59,0.08,4.4,0.086,6,29,0.9974,3.38,0.5,9] -- 4
#[8.6,0.42,0.39,1.8,0.068,6,12,0.99516,3.35,0.69,11.7] -- 8
x = np.array([[7.4,0.59,0.08,4.4,0.086,6,29,0.9974,3.38,0.5,9],[8.6,0.42,0.39,1.8,0.068,6,12,0.99516,3.35,0.69,11.7]])
x_scaled = feature_scaling_normalization(x , "predict" , xMeans, xMax, xMin)
y_predict = predict(theta, x_scaled)
print y_predict
