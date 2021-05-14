## Introduction

Numpy is a main package that has been popularized for many years. In deep learning and machine learning fields, we are using Numpy for some functionality like np.exp, np.log and np.reshape. 

This exercise is for some functions widely used in deep learning area, we will discuss some aspects on sigmoic function, graident descent of sigmoid, derivative computation, image vectorization, normalization and regularization as well as optimazation.

## 1. Basic functions

### 1.1 Sigmoid function

Sigmoid function is the function that we also name as the logistic function. It is a non-linear function used very well in machine learning and deep learning area as a classifer. 

Let's see its formula and image:

![image](https://user-images.githubusercontent.com/71245576/118286803-d9656500-b4a0-11eb-8f77-3d5a312e7b37.png)

The formula is: sigmoid(x) = 1/(1+e^-x), we can munually code for this function like this:

```python
import math
from public_tests import *

def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s
```

For example, want to know the basic_sigmoid(1):

```python
print("basic_sigmoid(1) = " + str(basic_sigmoid(1)))

basic_sigmoid_test(basic_sigmoid)
```

Note that "math" library in deep learning is rerely used because in deep learning we have to use matrices and vectors instead of real numbers.

when we run our manually coded function basic_sigmoid(x), when x is a vector:
```python
x = [1, 2, 3] # x becomes a python list object
basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
```
It sends a exception saying that bad operand type for unary -: 'list'

But we can use numpy to compute the vector and arrays:

```python
import numpy as np

# example of np.exp
t_x = np.array([1, 2, 3])
print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))
```
When if x is a vector, then a Python operation such as s=x+3 or s=1/x will output s as a vectpr of the same size as x.

Now let's use numpy package to compute sigmoid values: we need to define sigmoid function and call functions:

```python
def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s
```

call the function sigmoid(x):
```python
t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

sigmoid_test(sigmoid)
```

The result:

![image](https://user-images.githubusercontent.com/71245576/118290542-8db4ba80-b4a4-11eb-86c3-05667795bf5e.png)

### 1.2 Sigmoid gradient

We sometimes need to compute gradients to optimize loss functions using backpropagation. Like for example, implementing the function sigmoid_grad() to compute the gradient of the sigmoid function. The derivative of the sigmoid function is:

![image](https://user-images.githubusercontent.com/71245576/118307469-bf378100-b4b8-11eb-9d72-927ed1bc8d3c.png)

Now, we code this gradient function in two steps: set s to be the sigmoid of x and compute its derivatives

```python
def sigmoid_derivative(x):
    
    s = sigmoid(x)
    ds = s*(1-s)

    
    return ds
```

Instance calls the function sigmoid_derivative(x):
```python
t_x = np.array([1, 2, 3])
print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))

sigmoid_derivative_test(sigmoid_derivative)
```

### 1.3 Reshaping arrays

The common numpy functions used in deep learning are np.shape and np.reshape(), the first one is used to get the dimension of a matrix or vector, the another is used to reshape a matrix or vector to other dimension.

For example, an image may be represented by a 3D array of shape: length, height and depth. When you read an image as the input you should convert 3D data to other shapes as a 1D vector.

Now, let's implement a convector to reshape an input of shape(length, height, 3) to a vector of shape(length*height*3, 1).
For example, reshape an array v of shape(a, b, c) into a vector of shape(a*b,c). 

Reshape a image of length, height and depth to 1D shape:
```python
def image2vector(image):

    v= image.reshape((image.shape[0] * image.shape[1] * image.shape[2],1))

    return v
```

The instance is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values

```python
t_image = np.array([[[ 0.67826139,  0.29380381],
                     [ 0.90714982,  0.52835647],
                     [ 0.4215251 ,  0.45017551]],

                   [[ 0.92814219,  0.96677647],
                    [ 0.85304703,  0.52351845],
                    [ 0.19981397,  0.27417313]],

                   [[ 0.60659855,  0.00533165],
                    [ 0.10820313,  0.49978937],
                    [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(t_image)))

image2vector_test(image2vector)
```

The result is:

![image](https://user-images.githubusercontent.com/71245576/118311054-64ecef00-b4bd-11eb-9e4d-27795e852b13.png)


### 1.4 Normalizing rows

Actually there are several specific techniques to normalize our data for relative requirement. For gradient descent, it would be better if we normalize our data before invloving into machine learning and deep learning.

For example, we mean changing x to x/||x||, dividing each row vector of x by its norm. Now let's normalize the rows of a matrix, there are steps: 

1. Compute the norm of x
2. Divide x by its norm

```python


def normalize_rows(x):
    x_norm = np.linalg.norm(x, ord =2, axis= 1, keepdims = True)
    x = x/x_norm

    return x
```

Instance:

```python
x = np.array([[0, 3, 4],
              [1, 6, 4]])
print("normalizeRows(x) = " + str(normalize_rows(x)))

normalizeRows_test(normalize_rows)
```

The result shows below:
```python
![image](https://user-images.githubusercontent.com/71245576/118317099-a2ee1100-b4c5-11eb-9421-be954be78a00.png)

```

### 1.5 Broadcasting

In normalize_rows(), you can try to print the shapes of x_norm and x, and then rerun the assessment. You'll find out that they have different shapes. This is normal given that x_norm takes the norm of each row of x. So x_norm has the same number of rows but only 1 column. So how did it work when you divided x by x_norm? This is called broadcasting and we'll talk about it now!

Softmax function is a normalizing function used when we want to classify two or more classes:

![image](https://user-images.githubusercontent.com/71245576/118317330-e8aad980-b4c5-11eb-9944-eed9f240b89c.png)

Now we make a softmax function and see how the broadcasting works here:

```python
def softmax(x):

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims= True)
    s=x_exp/x_sum
    
    return s
```

Instance:

```python
t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(t_x)))

softmax_test(softmax)
```
If you print the shapes of x_exp, x_sum and s above and rerun the assessment cell, you will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). x_exp/x_sum works due to python broadcasting.

The result shows:

![image](https://user-images.githubusercontent.com/71245576/118318143-f745c080-b4c6-11eb-950b-6a7c8326a364.png)

## 2. Vectorization


### 2.1 Implement dot, outer, elementwise product

In deep learning, you deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run. To make sure that your code is computationally efficient, you will use vectorization.

For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.
```python
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1), len(x2))) # we create a len(x1)*len(x2) matrix with only zeros

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i] * x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
```

The result: different types of vectorization and its time consumed:

![image](https://user-images.githubusercontent.com/71245576/118318379-4f7cc280-b4c7-11eb-9ebb-bb877f89ae3a.png)

Try use built-in functions in numpy and test for the time it would be taken:
```python
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
```

The result:

![image](https://user-images.githubusercontent.com/71245576/118318555-8ce15000-b4c7-11eb-96f6-755882096762.png)

The vectorized implementation is much cleaner and more efficient. For bigger vectors and matrices, the differences would be even bigger.

Note that np.dot() performs a matrix-matrix or matrix-vector multiplication. This is different from np.multiply() and the * operator (which is equivalent to .* in Matlab/Octave), which performs an element-wise multiplication.

### 2.2 Implement the L1 and L2 loss functions

The loss is used to evaluate the performance of the model. The bigger the loss is, the more different the pre3dictions of the label are from the true values. In deep learning so you use optimization algorithms like gradient descent to minimize the cost.

L1 loss is defined as:

![image](https://user-images.githubusercontent.com/71245576/118319002-2577d000-b4c8-11eb-85d7-7fc06b4a3b76.png)

```python
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(abs(yhat-y))
    
    return loss
 ```
 
Instance:
 
 ```python
 yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

L1_test(L1)
```

The L1 loss here is 1.1. Now let's look at L2 loss: which is defined as:

![image](https://user-images.githubusercontent.com/71245576/118319524-e5651d00-b4c8-11eb-95d7-fceefd9ed600.png)

Define the L2 regression:
```python
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.dot(yhat-y,yhat-y)
    
    
    return loss
 ```
 Instance:
 
 ```python
 yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat, y)))

L2_test(L2)
```

L2 is 0.43.

## Reference:

Python Basics with Numpy, retrieved from https://www.coursera.org/learn/neural-networks-deep-learning/programming/isoAV/python-basics-with-numpy/lab






