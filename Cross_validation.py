#coding: utf-8
import random
import numpy as np

#目的変数１つ用の交差検定関数,引数のfuncにはscikit-learnのモデルクラスを想定
def cross_validation(func, data, times=10):
    samples = random.sample(len(data))
    newx = samples[:, :-1]
    newy = samples[:, -1:]

    sum=0
    for i in range(times):
        accuracy=0
        print("%d of %d" % (i, times))
        crossnum = np.array([False] * len(data))
        crossnum[i * len(data) / 10:(i + 1) * len(data) / 10] = True
        func.fit(newx[crossnum],newy[crossnum])

        for j in func.predict(newx[~crossnum]):
            if j==newy[~crossnum]:
                accuracy+=1
        sum+=accuracy/len(newy[~crossnum])
    return sum / times
