from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    for n in range(N):
        scores = X[n] @ W # dim C
        scores -= np.max(scores)
        loss += -scores[y[n]] + np.log(np.sum(np.exp(scores)))
        
        softmax_outs = np.exp(scores) / np.sum(np.exp(scores))
        for j in range(C):
            dW[:,j] += softmax_outs[j] * X[n]
        dW[:,y[n]] -= X[n]
        
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= N
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    N_scores = X @ W # N x C matrix
    N_scores -= np.max(N_scores, axis=1, keepdims=True)
    N_numerators = np.exp(N_scores) # N x C
    N_denoms = np.sum(np.exp(N_scores), axis=1) # 1 x N 
    loss += np.sum(-np.log(N_numerators[np.arange(0,num_train),y] / N_denoms))
    
    N_softmax_out = N_numerators / N_denoms.reshape((-1,1)) # N x C
    temp = N_softmax_out
    temp[np.arange(num_train),y] -= 1 
    dW = X.T @ temp
        
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
