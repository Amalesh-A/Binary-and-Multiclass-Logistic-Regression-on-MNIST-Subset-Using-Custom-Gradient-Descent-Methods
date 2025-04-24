import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for i in range(self.max_iter):
            gradient = np.zeros(n_features)
            for j in range(n_samples):
                gradient += self._gradient(X[j], y[j])
            gradient /= n_samples
            self.W -= self.learning_rate * gradient
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = np.zeros(n_features)
                for i in range(len(X_batch)):
                    grad += self._gradient(X_batch[i], y_batch[i])
                grad /= len(X_batch)

                self.W -= self.learning_rate * grad

        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in indices:
                grad = self._gradient(X[i], y[i])
                self.W -= self.learning_rate * grad

        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        z = np.dot(self.W, _x)
        sigmoid = 1 / (1 + np.exp(_y * z))
        grad = -_y * sigmoid * _x
        return grad

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        z = X @ self.W
        prob_pos = 1 / (1 + np.exp(-z))
        prob_neg = 1 - prob_pos
        return np.vstack((prob_neg, prob_pos)).T

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        z = X @ self.W
        prediction = np.where(z >= 0, 1, -1)
        return prediction

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        prediction = self.predict(X)
        accuracy = np.mean(prediction == y)
        return accuracy
    
    def assign_weights(self, weights):
        self.W = weights
        return self