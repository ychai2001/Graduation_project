from scipy.io import loadmat
import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np
import cupy as cp
import time
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg

def cal_time(file):

    content = loadmat(file)
    #print(content.keys())

    a = content['A']
    #print(a.shape[0])

    n = a.shape[0]

    b = np.zeros(n)

    a = ssp.csr_matrix(a)
    A = csp.csr_matrix(a)
    B = cp.asarray(b)   # 이거는 cpu to gpu같은거

    t_cpu= 0
    t_gpu = 0

    for i in range(10):
        ts_cpu = time.time()
        sol_cpu = ssp_linalg.cg(a, b)[0]
        te_cpu = time.time()
        t_cpu += te_cpu-ts_cpu

    for j in range(10):
        ts_gpu = time.time()
        sol_gpu = csp_linalg.cg(A,B)[0].get()
        te_gpu = time.time()
        t_gpu += te_gpu-ts_gpu

    diff = np.linalg.norm(sol_cpu - sol_gpu)
    if diff > 1e-5:
        print(f"Solution doesn't match : {diff}")
    else:
        print(f'cpu 계산시간 = {t_cpu/10}')
        print(f'gpu 계산시간 = {t_gpu/10}')


# data = 'ex10.mat'
# cal_time(data)