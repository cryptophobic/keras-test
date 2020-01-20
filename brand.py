import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy
from sklearn import preprocessing

col_types = {'category': int, 'name': str}

icecat_dataset = pd.read_csv('/home/dima/PycharmProjects/TensorFlow/result_stier.csv', dtype=col_types)

# icecat_dataset = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(icecat_data), columns=icecat_data.columns)
#
y = icecat_dataset.category
X = icecat_dataset.drop('category', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

knn = KNeighborsClassifier(n_neighbors=1)

le = preprocessing.LabelEncoder()
le.fit(X_train.astype(str))
X_train = le.transform(X_train.astype(str))
X_test = le.transform(X_test.astype(str))

knn.fit(X_train, y_train)
#
# X_new = numpy.array([['asus']])
# print("форма массива X_new: {}".format(X_new.shape))
#
# prediction = knn.predict(X_new)
# print("Прогноз: {}".format(prediction))
# # print("Спрогнозированная метка: {}".format(icecat_dataset['category'][prediction]))
# #
# # y_pred = knn.predict(X_test)
# # print("Прогнозы для тестового набора:\n {}".format(y_pred))
# #
# # print("Правильность на тестовом наборе: {:.2f}".format(numpy.mean(y_pred == y_test)))
# #
# # print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))
# #
