# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import matlab.engine
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LinearRegression as LR


def MB_Learner(datafile, target):
    eng = matlab.engine.start_matlab()
    parents_index, children_index, spouse_index, undirect_index = [], [], [], []
    result = []
    p, c, u = eng.LocalStructure_Learner(datafile, target, nargout=3)
    if isinstance(p, float):
        parents_index.append(int(p))
    else:
        parents_index = [int(x) for item in p for x in item]
    if isinstance(c, float):
        children_index.append(int(c))
    else:
        children_index = [int(x) for item in c for x in item]
    if isinstance(u, float):
        undirect_index.append(int(u))
    else:
        undirect_index = [int(x) for item in u for x in item]
    for ci in children_index:
        p, c, u = eng.LocalStructure_Learner(datafile, ci, nargout=3)
        if isinstance(p, float):
            spouse_index.append(int(p))
        else:
            temp = [int(x) for item in p for x in item]
            spouse_index = spouse_index + temp
        undirect_index = []
        if isinstance(u, float):
            undirect_index.append(int(u))
        else:
            undirect_index = [int(x) for item in u for x in item]
    parents_index = (np.array(parents_index) - 1).tolist()
    children_index = (np.array(children_index) - 1).tolist()
    spouse_index = (np.array(spouse_index) - 1).tolist()
    for item in parents_index:
        result.append(("X" + str(item)))
    for item in spouse_index:
        result.append(("X" + str(item)))
    for item in children_index:
        result.append(("X" + str(item)))
    return result


def Double_MB_Learner(data_file, mb):
    d_mb = []
    index = []
    for m in mb:
        temp = int(m[1:]) + 1
        index.append(temp)
    for i in index:
        result = MB_Learner(data_file, i)
        for r in result:
            d_mb.append(r)
    d_mb = list(set(d_mb))
    return d_mb


def causal_effect_v1(datafile, train_data, p, gp):
    abnormal_node = "X" + str(train_data.shape[1] - 1)
    if abnormal_node in p:
        p.remove(abnormal_node)
    if abnormal_node in gp:
        gp.remove(abnormal_node)
    eng = matlab.engine.start_matlab()
    total_p = list(set(p + gp))
    result = {}
    for tp in total_p:
        tp_index = int(tp[1:]) + 1
        parents_index, parents = [], []
        p, c, u = eng.LocalStructure_Learner(datafile, tp_index, nargout=3)
        if isinstance(p, float):
            parents_index.append(int(p))
        else:
            parents_index = [int(x) for item in p for x in item]
        parents_index = (np.array(parents_index) - 1).tolist()
        for pi in parents_index:
            parents.append(("X" + str(pi)))
        if len(parents) > 17:
            parents.insert(0, tp)
            if abnormal_node in parents:
                parents.remove(abnormal_node)
                parents.append("Y")
            dependent = parents
            X = train_data[dependent]
            Y = train_data["Y"]
            reg = LR().fit(X, Y)
            coef = reg.coef_
            ce = coef[0]
            result[tp] = abs(ce)
            continue
        if abnormal_node in parents:
            parents.remove(abnormal_node)
            parents.append("Y")
        ce = discrete_causal_effect(train_data, tp, parents)
        result[tp] = abs(ce)
    return result


def discrete_causal_effect(data, X, Z):
    k = len(Z)
    n = len(data)
    ace = 0
    if len(Z) == 0:
        p1 = (len(data.loc[(data[X] == 1) & (data["Y"] == 1)])) / (len(data.loc[data[X] == 1]))
        p0 = (len(data.loc[(data[X] == 0) & (data["Y"] == 1)])) / (len(data.loc[data[X] == 0]))
        ace = p1 - p0
    else:
        p1 = 0
        for item in product([0, 1], repeat=k):
            d = data
            for i in range(k):
                d = d[d[Z[i]] == item[i]]
            d_f = d[(d["Y"] == 1) & (d[X] == 1)]
            d_b = d[d[X] == 1]
            if len(d_b) == 0:
                p1 += 0
            else:
                p1 += (len(d_f) / len(d_b)) * (len(d) / n)
        p0 = 0
        for item in product([0, 1], repeat=k):
            d = data
            for i in range(k):
                d = d[d[Z[i]] == item[i]]
            d_f = d[(d["Y"] == 1) & (d[X] == 0)]
            d_b = d[d[X] == 0]
            if len(d_b) == 0:
                p0 += 0
            else:
                p0 += (len(d_f) / len(d_b)) * (len(d) / n)
        ace = p1 - p0
    return ace


def order_dict(dict, k):
    d_order = sorted(dict.items(), key=lambda item: (-item[1], item[0]))
    count = 0
    result = []
    for d in d_order:
        if count == k:
            break
        result.append(d[0])
        count += 1
    return result


if __name__ == '__main__':
    bias = 0.25
    p = 20
    data_file = "train_biasrate=" + str(bias) + ".csv"
    train_data = pd.read_csv("dataset/Dgbr_data/bias_rate=0.25_p=20/" + data_file, delimiter=",")
    fea_list = train_data.columns.tolist()
    causal_flag = True
    if causal_flag:
        k = 6
        mb = MB_Learner(data_file, len(fea_list))
        mb = list(set(mb))
        d_mb = Double_MB_Learner(data_file, mb)
        d_mb = list(set(d_mb))
        ce_result = causal_effect_v1(data_file, train_data, mb, d_mb)
        ce_result_k = order_dict(ce_result, k)
        ce_result_k.append("Y")
        train_data = train_data[ce_result_k]
    X_train, Y_train = train_data.drop(columns=["Y"]), train_data["Y"]
    clf = SVC(probability=True)
    clf.fit(X_train, Y_train)
    train_score = clf.score(X_train, Y_train)
    score_list = []
    rmse_list = []
    for base in range(1, 10):
        filename = "test_biasrate=" + str(round(base * 0.1, 1)) + ".csv"
        test_data = pd.read_csv("dataset/Dgbr_data/bias_rate=0.25_p=20/" + filename, delimiter=",")
        if causal_flag:
            test_data = test_data[ce_result_k]
        X_test, Y_test = test_data.drop(columns=["Y"]), test_data["Y"]
        score = clf.score(X_test, Y_test)
        score_list.append(score)
        Y_prob = clf.predict_proba(X_test)[:, 1]
        rmse = np.sqrt(np.mean((Y_test - Y_prob) ** 2))
        rmse_list.append(rmse)
    print(rmse_list)
    print("The average rmse is ：", np.mean(rmse_list))
