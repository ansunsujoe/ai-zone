from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# Matrix square
def vector_square(X):
    new_vector = np.zeros(shape=X.shape)
    for i in range(len(X)):
        new_vector[i] = X[i] ** 2

# Create custom kernels
def quad_kernel(X, Y):
    # squared_X = vector_square(X)
    # squared_Y = vector_square(Y)
    return (np.dot(X, Y.T)) ** 2

# Create the data
X_train = np.array([
    [2, 2], [3, 2.5], [4.25, 2.5], [4.7, 2.3], [4.9, 2],
    [3.5, 3.6], [3.4, 4.3], [3.3, 5.9], [4.3, 3.9], [5.5, 8],
    [8, 5.2], [6.2, 6.4], [8.5, 7], [8.4, 7.8], [7.5, 8.4],
    [7.4, 8.7], [7.7, 9.4], [7.6, 9.6], [9.2, 9.2], [9.8, 8]
    ])
y_train = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Create a svm Classifier
clf = svm.SVC(kernel=quad_kernel) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

# create a mesh to plot in
h = 0.02

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training points
# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()