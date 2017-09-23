import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    wx = X[i].dot(W)
    wx -= np.max(wx) #normalized wx
    S = np.exp(wx)/np.sum(np.exp(wx)) #softmax
    for j in xrange(num_classes):
      yi=0
      if(j == y[i]):
        loss -= np.log(S[j])
        yi=1
      smy = S[j] - yi # scalar / number
      dW[:,j] += smy * X[i]
  
  loss = loss / float(num_train)
  dW = dW / float(num_train)

  loss += reg * np.sum(W*W)
  dW += 2 * reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  WX = X.dot(W)
  WX -= np.max(WX, axis=1, keepdims=True)
  S = np.exp(WX)/np.sum(np.exp(WX),axis=1,keepdims=True)

  loss = -np.sum(np.log(S[np.arange(num_train),y]))
  loss /= float(num_train)
  loss += reg * np.sum(W * W)  
    
  S[np.arange(num_train),y] -= 1 #dL/dS derviate of Loss function wrt S 
  dW = X.T.dot(S) # dS/dW = X.T; dL/dW = dL/dS * ds/dW = X.T.dot(S)
  dW /= float(num_train)
  dW += 2 * reg * W 
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

