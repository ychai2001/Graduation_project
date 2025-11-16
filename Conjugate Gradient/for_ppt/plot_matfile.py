from scipy.io import loadmat
import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np
import cupy as cp
import time
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg


def make_b(A, x_true=None):

    n = A.shape[0]

    if x_true == None:
        x_true = np.random.rand(n).astype(np.float32)
        b = A.dot(x_true).astype(np.float32)
    else:
        b = A.dot(x_true).astype(np.float32)

    return b, x_true


def cal_time(file, x_true=None):

    content = loadmat(file)

    a = content['A']
    n = a.shape[0]
    b, x = make_b(a, x_true)

    a = ssp.csr_matrix(a, dtype=np.float32)
    A = csp.csr_matrix(a, dtype=np.float32)
    B = cp.asarray(b, dtype=np.float32)   # 이거는 cpu to gpu같은거

    t_cpu= 0
    t_gpu = 0

    for i in range(5):
        ts_cpu = time.time()
        sol_cpu = ssp_linalg.cg(a, b)[0]
        te_cpu = time.time()
        t_cpu += te_cpu-ts_cpu

    for j in range(5):
        ts_gpu = time.time()
        sol_gpu = csp_linalg.cg(A, B)[0].get()
        te_gpu = time.time()
        t_gpu += te_gpu-ts_gpu

    diff = np.allclose(sol_cpu, sol_gpu)
    print(f'Exact solution = {x}\n')
    print(f'CPU solution RMSE Error = {np.linalg.norm(sol_cpu - x)}')
    print(f'GPU solution RMSE Error = {np.linalg.norm(sol_gpu - x)}\n')
    if diff > 1e-5:
        print(f"Solution doesn't match : {diff}")
    else:
        print(f'cpu 계산시간 = {t_cpu/5}(s)')
        print(f'gpu 계산시간 = {t_gpu/5}(s)\n')
        print(f'GPU 성능은 CPU의 {t_cpu/t_gpu}배')

