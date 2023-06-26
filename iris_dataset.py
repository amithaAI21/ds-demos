from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)
print(iris.DESCR)   # each 50 data for setosa,verginica,versicolor(3 flower types)
print(iris.data.shape)
print("Label encoding:",iris.target) # label encoding converts textual data to numerical
# for better understanding for computers(0-versicolor,1-setosa,2-verginica)