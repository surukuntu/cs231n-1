import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
    
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  Q = np.zeros((num_train, num_classes))
  loss = 0.0
  for i in xrange(num_train):
    dDot = np.zeros(num_classes) 
    scores = X[i].dot(W) #1x3073 dot 3073x10 = 1x10
    xi = X[i, :]
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        Q[i,j] = 1
        Q[i,y[i]] -= 1
        dW[:,j] += xi
        dW[:,y[i]] -= xi

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  #add regularization derivative
  dW += reg * 2 * W
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

 # print("X shape : ", X.shape, ", W shape :", W.shape, ", dDot shape", dDot.shape, "dW shape", dW.shape)
  #print("dDot = ", dDot[:10])
  #print(Q[:10,:])

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #dW = X.T.dot(dDot) 
  #print("dW = ", dW[:10])  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  n = X.shape[0]
  wx = X.dot(W) #calculate wx
  wyix = wx[np.arange(n),y,None] #extract all the predictions for the correct class
  P = wx - wyix + 1 #calculate loss
  P[np.arange(n),y] = 0 #set correct class prediction to zero
  P = np.maximum(0,P) # min threshold to zero
  loss = np.sum(P)/n
  reg_loss = reg * np.sum(W*W)
  tot_loss = loss +reg_loss
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  P[P>0] = 1 
  P[np.arange(n),y] = -np.sum(P, axis=1)
  #print(P[:10,:])
  dW = (X.T.dot(P))/n + reg * 2 * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
