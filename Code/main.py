import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "D:/TAMU Coursework/Deep Learning/HW1/data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    plt.figure(figsize=(8, 6))
    for label, marker, color in zip([-1, 1], ['x', 'o'], ['red', 'blue']):
        plt.scatter(X[y == label, 0], X[y == label, 1], marker=marker, color=color, label=label)
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Training Features Visualization")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_features.png")  # You can also save as .pdf if needed
    plt.close()

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    plt.figure(figsize=(8, 6))
    for label, marker, color in zip([-1, 1], ['x', 'o'], ['red', 'blue']):
        plt.scatter(X[y == label, 0], X[y == label, 1], marker=marker, color=color, label=f'Class {label}')

    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(W[0] + W[1] * x_vals) / W[2]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Decision Boundary - Sigmoid Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_result_sigmoid.png")
    plt.close()
	
def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(8, 6))

    # Create a meshgrid over the feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Flatten and add bias term (ones)
    grid = np.c_[xx.ravel(), yy.ravel()]
    bias = np.ones((grid.shape[0], 1))
    grid_features = np.hstack((bias, grid))  # shape [num_points, 3]

    # Compute class scores and predictions
    Z = grid_features @ W  # logits
    Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # softmax trick
    Z = Z / np.sum(Z, axis=1, keepdims=True)
    preds = np.argmax(Z, axis=1)
    preds = preds.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, preds, alpha=0.3, cmap='coolwarm')

    # Plot training data
    markers = ['o', 's', 'x']
    colors = ['red', 'green', 'blue']
    for i in range(3):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    c=colors[i], marker=markers[i], label=f'Class {i}', edgecolors='k')

    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Decision Boundaries - Multiclass Softmax")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_result_softmax.png")
    plt.close()

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    #logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    #logisticR_classifier.fit_BGD(train_X, train_y)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))
    #accuracyBGD = logisticR_classifier.score(train_X, train_y)
    #print(f"Accuracy of BGD: {accuracyBGD * 100:.2f}%")
    #print("\n")

    #visualize_result(train_X[:, 1:3], train_y, logisticR_classifier.get_params())

    #logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    #logisticR_classifier.fit_miniBGD(train_X, train_y, 100)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))
    #accuracyminiBGD = logisticR_classifier.score(train_X, train_y)
    #print(f"Accuracy of miniBGD: {accuracyminiBGD * 100:.2f}%")
    #print("\n")

    #logisticR_classifier.fit_SGD(train_X, train_y)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))
    #accuracySGD = logisticR_classifier.score(train_X, train_y)
    #print(f"Accuracy of SGD: {accuracySGD * 100:.2f}%")
    #print('\n')

    #probs = logisticR_classifier.predict_proba(train_X)
    #print("Predict Probability:\n", probs[:5]) 

    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all



    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    accuracy = logisticR_classifier_multiclass.score(train_X, train_y)
    print(f"Accuracy of multiclass softmax classifier on training data: {accuracy * 100:.2f}%")
    print('\n')
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass.get_params())


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


'''Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
'''
'''
    ### YOUR CODE HERE

    ### END YOUR CODE

    # ------------End------------
'''
    

if __name__ == '__main__':
	main()