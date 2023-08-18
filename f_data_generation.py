# -*- coding: utf-8 -*-
import numpy as np
import math


def data_generation_s2v(n, p, p_causal, p_causal_f, p_bias, bias_rate):
    n_bak = n
    n = n * (2 ** p_bias) * 2

    X = np.zeros([n, p])
    beta = np.zeros([p, 1])
    # Y = np.random.normal(0,0,[n,1])
    for i in range(p):
        # np.random.seed(seed+(i+1))
        rand = np.random.normal(0, 1, [n, 1])
        rand[rand > 0] = 1
        rand[rand <= 0] = 0
        X[:, i:i + 1] = np.copy(rand)
        if (i < p_causal):
            print(i)
            beta[i] = ((i % 3 + 1) * p / 3) * (-1) ** (i)
        '''
        if ((i+1) % mod == 0):
            #Y += X[:,i:i+1]
            beta[i] = (i-p/2)*(-1)**((i+1)/mod)
            #beta[i] = p/4*(-1)**((i+1)/mod)
        '''
    C_X = np.copy(X)
    C_beta = np.copy(beta)
    for i in range(p_causal + 1, p_causal + p_causal_f + 1):
        C_X = np.hstack((C_X, X[:, i - 1:i] * X[:, i:i + 1]))
        C_beta = np.vstack((C_beta, p / 2))
    for i in range(p - p_bias, p):
        temp = np.random.normal(0, 2, [n, 1]) + X[:, i - p + p_bias:i - p + p_bias + 1] + X[:,
                                                                                          i - p + p_bias + 1:i - p + p_bias + 2]  # np.sum(X[:,i-p+p_bias:i-p+p_bias+5],1)
        temp[temp <= 1] = 0
        temp[temp > 1] = 1
        temp = np.reshape(temp, (n, 1))
        X[:, i:i + 1] = np.copy(temp)
    noise = np.random.normal(0, 0.2, [n, 1])
    # Y_true = 1/(1+math.e**(np.matmul(-X, beta)))
    Y_temp = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta))) + noise
    Y = np.zeros([n, 1])
    Y[Y_temp >= 0.5] = 1
    # selection bias
    Y_compare = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta)))
    Y_compare[Y_compare >= 0.5] = 1
    Y_compare[Y_compare < 0.5] = 0
    index_pre = np.ones([n, 1], dtype=bool)
    for i in range(p - p_bias, p):
        # np.random.seed(seed+100+i+1)
        selection_bias = np.random.random([n, 1])
        index_pre = index_pre & (
                    (X[:, i:i + 1] == Y_compare) & (selection_bias <= bias_rate) | (X[:, i:i + 1] != Y_compare) & (
                        selection_bias >= bias_rate))
    index = np.where(index_pre == True)
    X_re = X[index[0], :]
    Y_re = Y[index[0]]
    return X_re[0:n_bak, :], Y_re[0:n_bak, :]


def data_generation(n, p, p_causal, p_causal_f, p_bias, bias_rate):
    X = np.zeros([n, p])
    beta = np.zeros([p, 1])
    # Y = np.random.normal(0,0,[n,1])
    for i in range(p):
        # np.random.seed(seed+(i+1))
        rand = np.random.normal(0, 1, [n, 1])
        rand[rand > 0] = 1
        rand[rand <= 0] = 0
        X[:, i:i + 1] = np.copy(rand)
        if (i < p_causal):
            # print(i)
            beta[i] = ((i % 3 + 1) * p / 3) * (-1) ** (i)
            # beta[i] = ((i % 3 + 1) ) * (-1) ** (i)

    C_X = np.copy(X)
    C_beta = np.copy(beta)

    for i in range(p_causal + 1, p_causal + p_causal_f + 1):
        # print(i)
        C_X = np.hstack((C_X, X[:, i - 1:i] * X[:, i:i + 1]))
        C_beta = np.vstack((C_beta, p / 2))

    noise = np.random.normal(0, 0.2, [n, 1])
    # Y_true = 1/(1+math.e**(np.matmul(-X, beta)))
    Y_temp = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta))) + noise
    Y = np.zeros([n, 1])
    Y[Y_temp >= 0.5] = 1

    # selection bias
    Y_compare = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta)))
    Y_compare[Y_compare >= 0.5] = 1
    Y_compare[Y_compare < 0.5] = 0
    for i in range(p - p_bias, p):
        # print(i)
        selection_bias = np.random.random([n, 1])
        temp = np.copy(X[:, i:1 + i])
        index_equal = np.where(selection_bias <= bias_rate)
        index_notequal = np.where(selection_bias > bias_rate)
        temp[index_equal] = Y_compare[index_equal]
        temp[index_notequal] = 1 - Y_compare[index_notequal]
        X[:, i:1 + i] = np.copy(temp)
        # print(np.corrcoef(X[:,i*mod],Y[:,0]))
    '''
    mod = 2
    Y_compare = 1/(1+math.e**(np.matmul(-C_X, C_beta)))
    Y_compare[Y_compare>=0.5] = 1
    Y_compare[Y_compare<0.5] = 0
    for i in range(p_bias):
        selection_bias = np.random.random([n,1])
        temp = np.copy(X[:,i*mod:1+i*mod])
        index_equal = np.where(selection_bias<=bias_rate)
        index_notequal = np.where(selection_bias>bias_rate)
        temp[index_equal] = Y_compare[index_equal]
        temp[index_notequal] = 1-Y_compare[index_notequal]
        X[:,i*mod:1+i*mod] = np.copy(temp)
        #print(np.corrcoef(X[:,i*mod],Y[:,0]))
    '''
    return X, Y


def data_generation_v2s(n, p, p_causal, p_causal_f, p_bias, bias_rate):
    n_bak = n
    n = n * (2 ** p_bias) * 2

    X = np.zeros([n, p])
    beta = np.zeros([p, 1])
    # Y = np.random.normal(0,0,[n,1])
    for i in range(p):
        # np.random.seed(seed+(i+1))
        rand = np.random.normal(0, 1, [n, 1])
        rand[rand > 0] = 1
        rand[rand <= 0] = 0
        X[:, i:i + 1] = np.copy(rand)
        if (i < p_causal):
            beta[i] = ((i % 3 + 1) * p / 3) * (-1) ** (i)
        # if (i < p_bias):
        #     beta[i] = ((i % 3 + 1) * p / 3) * (-1) ** (i)
        '''
        if ((i+1) % mod == 0):
            #Y += X[:,i:i+1]
            beta[i] = (i-p/2)*(-1)**((i+1)/mod)
            #beta[i] = p/4*(-1)**((i+1)/mod)
        '''

    for i in range(p_causal):
        temp = np.random.normal(0, 2, [n, 1]) + X[:, p - p_bias + i:p - p_bias + i + 1] + X[:,
                                                                                          p - p_bias + i + 1:p - p_bias + i + 2]  # np.sum(X[:,i-p+p_bias:i-p+p_bias+5],1)
        temp[temp <= 1] = 0
        temp[temp > 1] = 1
        temp = np.reshape(temp, (n, 1))
        X[:, i:i + 1] = np.copy(temp)
        beta[i] = ((i % 3 + 1) * p / 3) * (-1) ** (i)

    C_X = np.copy(X)
    C_beta = np.copy(beta)

    for i in range(p_causal + 1, p_causal + p_causal_f + 1):
        C_X = np.hstack((C_X, X[:, i - 1:i] * X[:, i:i + 1]))
        C_beta = np.vstack((C_beta, p / 2))
    noise = np.random.normal(0, 0.2, [n, 1])
    # Y_true = 1/(1+math.e**(np.matmul(-X, beta)))
    Y_temp = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta))) + noise
    Y = np.zeros([n, 1])
    Y[Y_temp >= 0.5] = 1

    # selection bias
    Y_compare = 1 / (1 + math.e ** (np.matmul(-C_X, C_beta)))
    Y_compare[Y_compare >= 0.5] = 1
    Y_compare[Y_compare < 0.5] = 0
    index_pre = np.ones([n, 1], dtype=bool)
    for i in range(p - p_bias, p):
        # np.random.seed(seed+100+i+1)
        selection_bias = np.random.random([n, 1])
        index_pre = index_pre & (
                (X[:, i:i + 1] == Y_compare) & (selection_bias <= bias_rate) | (X[:, i:i + 1] != Y_compare) & (
                selection_bias >= bias_rate))
    index = np.where(index_pre == True)
    X_re = X[index[0], :]
    Y_re = Y[index[0]]

    return X_re[0:n_bak, :], Y_re[0:n_bak, :]
