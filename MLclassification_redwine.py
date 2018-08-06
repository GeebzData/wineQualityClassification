
# import general libraries
import pandas as pd
import numpy as np

# import visualization libraries
import matplotlib.pyplot as plt
%matplotlib inline

# import sklearn model libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics

# import white wine review csv file
url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data_red = pd.read_csv(url, sep = ';')

# create a list of feature names from data_red to help in creating feature data
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# use the list to select a subset of the original DataFrame
X_red = data_red[feature_names]

# target data from the quality column
y_red = data_red.quality

# Split into training and test set
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_red, y_red, test_size = 0.2, random_state=42, stratify=y_red)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train_r, y_train_r)

# calculate and print the accuracy
x = knn.score(X_test_r, y_test_r)

print("The accuracy of this model is: " + str(round(x*100,2)) + "%")
if x > .75:
    print("This model is pretty good")
else:
    print("This model is not accurate and there probably is not a good way to determine wine quality based on data of wine characteristics")

# Setup arrays to store train and test accuracies based on 1-9 n_neighbors
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train_r, y_train_r)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train_r, y_train_r)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test_r, y_test_r)

# generate duel line graph for Testing Accuracy and Training Accuracy pending n_neighbors value
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
