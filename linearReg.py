# UserName : Steve Hu
# Data : 2017-10-16
# Description : test linear regression model

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)

from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data
y = boston.target

# divide the datasets into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

import tensorflow as tf
n_features = X_train.shape[1]  # 
n_samples = X_train.shape[0]

