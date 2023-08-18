# -*- coding: utf-8 -*-
from sklearn.linear_model import LassoCV
import matlab.engine
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from f_data_generation import data_generation_v2s


def data_transformer():
    fileList = ["amazon.txt", "Caltech10.txt", "dslr.txt", "webcam.txt"]
    for file in fileList:
        columns = []
        fileLocation = "dataset/Image_data/" + file
        data = np.loadtxt(fileLocation, delimiter="\t")
        d = data.shape[1]
        for i in range(d - 1):
            columns.append("X" + str(i))
        columns.append("Y")
        data_new = pd.DataFrame(data=data, columns=columns)
        fileLocation_new = "dataset/Image_data/" + file[:-3] + "csv"
        data_new.to_csv(fileLocation_new, index=False)


def lasso_selection(source_data, k):
    s_data = shuffle(source_data, random_state=1)
    Y = s_data["Y"]
    X = s_data.drop(columns=["Y"])
    LC = LassoCV(cv=5, copy_X=True, random_state=42).fit(X, Y)
    cof = pd.Series(LC.coef_, copy=True, index=X.columns)
    new_cof = abs(cof.reindex(cof.abs().sort_values(ascending=False).index))
    count = 0
    for item in new_cof:
        if item != 0:
            count += 1
        print(item)
    selected_features = new_cof.index.to_list()[:k]
    return selected_features


def feature_selection(data, dataname):
    total_fea = data.columns.tolist()
    k = 50
    select_fea = lasso_selection(data, k)
    for f in total_fea:
        if f not in select_fea and f != "Y":
            data = data.drop(columns=[f])
    s_data = shuffle(data, random_state=1)
    s_data_name = dataname[:-4] + "_" + str(k) + "features" + ".csv"
    file_location = "Amazon_sub_data/" + s_data_name
    s_data.to_csv(file_location, index=False)
    return s_data_name, s_data


def global_dag_learning(filename):
    eng = matlab.engine.start_matlab()
    mat_result = eng.GlobalStructure_learner(filename)
    print(mat_result)
    print(type(mat_result))
    dag = np.array(mat_result)
    result = pd.DataFrame(dag)
    result.to_csv("Result/DAG_Learning_result/" + filename[:-4] + "_dag(gsbn).csv", index=False)
    return dag


def search_parents(dag, feature_list, flag):
    d = dag.shape[1]
    p_index = []
    gp_index = []
    parents = []
    g_parents = []
    for i in range(d):
        if dag[i][d - 1] == 1:
            p_index.append(i)
    for p in p_index:
        parents.append(feature_list[p])
    if flag == True:
        for p in p_index:
            for j in range(d):
                if dag[j][p] == 1:
                    gp_index.append(j)
        for gp in gp_index:
            g_parents.append(feature_list[gp])
    return parents, g_parents


def data_generator(train_flag, bias_rate, p_input):
    n_input = 5000
    p_bias = int(p_input * 0.4)
    p_causal = int(p_input * 0.3)
    p_causal_f = int(p_input * 0.2)
    X_in, Y_in = data_generation_v2s(n_input, p_input, p_causal, p_causal_f, p_bias, bias_rate)
    columns = []
    for i in range(p_input):
        columns.append("X" + str(i))
    columns.append("Y")
    XY_in = pd.DataFrame(data=np.concatenate((X_in, Y_in), axis=1), columns=columns)
    if train_flag:
        XY_in.to_csv("dataset/Dgbr_data/" + "train_biasrate=" + str(bias_rate) + ".csv", index=False)
    else:
        XY_in.to_csv("dataset/Dgbr_data/" + "test_biasrate=" + str(bias_rate) + ".csv", index=False)
