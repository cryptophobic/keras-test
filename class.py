from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy

iris_dataset = load_iris()
# print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'][:193] + "\n...")
# print("Названия ответов: {}".format(iris_dataset['target_names']))
# print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
# print("Тип массива data: {}".format(type(iris_dataset['data'])))
# print("Форма массива data: {}".format(iris_dataset['data'].shape))
# print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
# print("Тип массива target: {}".format(type(iris_dataset['target'])))
# print("Форма массива target: {}".format(iris_dataset['target'].shape))
# print("Ответы:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# # создаем dataframe из данных в массиве X_train
# # маркируем столбцы, используя строки в iris_dataset.feature_names
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# # создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train
# grr = scatter_matrix(
#     iris_dataframe,
#     c=y_train,
#     figsize=(15, 15),
#     marker='o',
#     hist_kwds={'bins': 20},
#     s=60,
#     alpha=.8,
#     cmap=mglearn.cm3
# )
#
# plt.show()


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = numpy.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))

print("Правильность на тестовом наборе: {:.2f}".format(numpy.mean(y_pred == y_test)))

print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))
