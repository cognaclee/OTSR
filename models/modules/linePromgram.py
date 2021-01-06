#-*- coding:utf-8 -*-
from pulp import LpProblem,LpMaximize,LpVariable,LpContinuous,lpSum
import numpy as np

def H_Star_Solution(fakeImg, trueImg, coef_K):
    fakeNum=fakeImg.shape[0]
    trueNum=trueImg.shape[0]
    transportCost=np.zeros((fakeNum,trueNum))

    for i in range(trueNum):
        for j in range(fakeNum):
            diff=(fakeImg[i]-trueImg[j]).numpy()
            square=diff*diff
            transportCost[i][j]=2*coef_K*np.sum(square)
    
    Ingredients1=['Y_'+str(i) for i in range(trueNum)]
    Ingredients2=['X_'+str(i) for i in range(fakeNum)]
    Ingredients=Ingredients1+Ingredients2

    costCoef = [1/trueNum for i in range(trueNum)]+[-1/fakeNum for j in range(fakeNum)]

    costs=dict(zip(Ingredients,costCoef))

    prob = LpProblem("H_Star_Solution", LpMaximize)

    ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0,1,LpContinuous)

    prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients])

    for i in range(trueNum):
        trueIndex='Y_'+str(i) 
        for j in range(fakeNum):
            fakeIndex='X_'+str(j)
            prob += lpSum([ingredient_vars[trueIndex],-ingredient_vars[fakeIndex]]) <= transportCost[i][j]

    prob.solve()

    HStar_real = [0]*trueNum
    Ord=0
    for i in Ingredients1:
        HStar_real[Ord] = ingredient_vars[i].value()
        Ord +=1
    
    HStar_fake = [0]*fakeNum
    Ord=0
    for i in Ingredients2:
        HStar_fake[Ord] = ingredient_vars[i].value()
        Ord +=1
    
    HStar_real = np.array(HStar_real)
    HStar_fake = np.array(HStar_fake)
    return HStar_real, HStar_fake

