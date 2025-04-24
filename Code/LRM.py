import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        n_samples, n_features = X.shape
        k = self.k
        self.W = np.zeros((n_features, k))

        y_onehot = np.zeros((n_samples, k))
        labels = labels.astype(int)
        y_onehot[np.arange(n_samples), labels] = 1


        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = np.zeros((n_features, k))
                for i in range(len(X_batch)):
                    grad += self._gradient(X_batch[i], y_batch[i])
                grad /= len(X_batch)

                self.W -= self.learning_rate * grad
        return self
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
        z = self.W.T @ _x
        probability = self.softmax(z)
        diff = probability - _y
        gradient = np.outer(_x, diff)
        return gradient
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        x_stable = x - np.max(x)
        exp_scores = np.exp(x_stable)
        return exp_scores / np.sum(exp_scores)
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        logits = X @ self.W
        probs = np.apply_along_axis(self.softmax, 1, logits)
        return np.argmax(probs, axis=1)

    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        preds = self.predict(X)
        return np.mean(preds == labels)