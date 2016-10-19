import numpy as np
import scipy
import os
import re
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# x is ( num_feature , num_example )
# y is ( num_example , 1 )
# theta is ( num_feature , 1)
# Don't have regularized term
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
        #print theta
    #Return theta, errorarr in different num_iteration and final minimum error(last element in errorarr)
    return theta,errorarr,errorarr[-1]

def chart_iteration_cost( if_regularized, x , y , theta,m , alpha , numIterations ):
    if if_regularized == "true":
        #Regularized Linear Regression
        theta, errorarr, min_error = gradientDescent2(x, y, theta, m, alpha, numIterations, 10)
    else:
        theta, errorarr, min_error = gradientDescent(x , y , theta , m , alpha , numIterations)
    plt.plot(range(1,numIterations+1),errorarr)
    plt.show()


#Hypothesis function has regularized term
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
        #print theta
    #Return theta, errorarr in different num_iteration and final minimum error(last element in errorarr)
    return theta,errorarr,errorarr[-1]


#Current file absolute path
cur_file=os.path.abspath(__file__)

#Directory moving pointers
moving_dir=os.path.dirname(cur_file)

#Find /data directory
while(not os.path.isdir(moving_dir + '/data')):
    moving_dir = os.path.dirname(moving_dir)
data_dir=moving_dir + '/data'

#Example files input filename and full path
x_file=data_dir+'/input.csv'
y_file=data_dir+'/y.csv'

#Check whether inputfile exists
if not os.path.isfile(x_file):
    raise Exception("Data example file %s is missing" %x_file )

#Initial Num of Feature and Num of samples/examples
num_feature=0
num_example=0

#Initial X (samples matrix)
x = None
with open(x_file,'r') as f:
    for i, row in enumerate(f):
        if x is not None:
            temparr = np.fromstring(row,dtype=float, sep = ',')
            if num_feature == temparr.shape[0]:
                x = np.vstack((x,temparr))
            else:
                raise Exception("Input samples have different feature dimensions")
        else:
            x = np.fromstring(row,dtype=float, sep = ',')
            num_feature = (x.shape)[0]
if x is not None:
    num_example=x.shape[0]
else:
    raise Exception("x (example matrix) is empty")

#Initialize Y (samples value)
y = None
#Read y from data directory
with open(y_file, 'r') as f:
    for row in f:
        #Catch the exception if there are non numeric character in y file
        try:
            float(row.rstrip())
        except ValueError:
            print "y file has non-numeric character"

        #Catch alphabet character
        if re.search( '[a-z][A-Z]]',row.rstrip()):
            raise Exception("y file has alphabet")

        if y is not None:
            y = np.append(y,float(row.rstrip()))
        else:
            y = np.array([float(row.rstrip())])

#Add dummy ones to matrix x for x0
x = np.insert(x, 0 , 1, axis = 1)

#Reshape x to normal shape and theta matrix
x = x.transpose()

#x is a matrix with dimension (num_feature+1, num_example)
#y is a vector with length num_example , y.shape() = (num_example,)
#theta is a matrix with dimension (num_feature+1,1) theta.shape() = (num_feature+1, 1)

#Convert y to (num_example,1) matrix
y= y.reshape(num_example,1)

#Random Choose theta
#theta = np.ones((num_feature+1,1))
theta = np.random.rand(num_feature+1,1)


#Call gradient descent function
#(theta,_,_)=gradientDescent(x, y , theta, num_example, 0.01 , 50000)
chart_iteration_cost( "true", x, y , theta, num_example , 0.001 , 100000)