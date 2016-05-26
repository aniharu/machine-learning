__author__ = 'umeco'

# coding:utf-8

from sklearn import tree
from sklearn.externals.six import StringIO
import csv
import numpy as np
import Cross_validation

#特定のデータ型の決定木図のpdfを作成する関数
def create_DecisionTree(data):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data[:,:-1], data[:,-1:])
    car_attribute=['buying','maint','doors','persons','lug_boot','safety']
    car_class=['unacc','acc','good','vgood']
    with open("car.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=car_attribute,
                         class_names=car_class,
                         filled=True, rounded=True,
                         special_characters=True)
    #graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #graph.write_pdf("car.pdf")

#車のデータを生成（String型では決定木を作成できない）
dic={'vhigh':4,'high':3,'med':2,'low':1,'5more':5,'more':5,
     'small':1,'big':3,'unacc':1,'acc':2,'good':3,'vgood':4,'2':2,'3':3,'4':4}

f=open("car.data", "r")
car_data=csv.reader(f)

car_data=[v for v in car_data]
for i in range(len(car_data)):
    for j in range(7):
        car_data[i][j] = dic[car_data[i][j]]

#整数型でデータを再生成
cdata=np.int32(car_data)

clf = tree.DecisionTreeClassifier()
print(str(Cross_validation.cross_validation(clf,cdata) * 100))

create_DecisionTree(cdata)