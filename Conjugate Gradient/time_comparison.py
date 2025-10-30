from scipy.io import loadmat
import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np
import cupy as cp
import time
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg

from matplotlib import pyplot as plt

def cal_time(file):

    content = loadmat(file)
    a = content['Problem'][0, 0]['A']

    n = a.shape[0]

    b = np.ones(n)

    a = ssp.csr_matrix(a)
    A = csp.csr_matrix(a)
    B = cp.asarray(b)   # 이거는 cpu to gpu같은거

    t_cpu= 0
    t_gpu = 0

    for i in range(5):
        ts_cpu = time.time()
        sol_cpu = ssp_linalg.cg(a, b)[0]
        te_cpu = time.time()
        t_cpu += te_cpu-ts_cpu

    for j in range(5):
        ts_gpu = time.time()
        sol_gpu = csp_linalg.cg(A,B)[0].get()
        te_gpu = time.time()
        t_gpu += te_gpu-ts_gpu

    diff = np.linalg.norm(sol_cpu - sol_gpu)
    if diff > 1e-5:
        print(f"Solution doesn't match : {diff}")
    else:
        print(f'cpu 계산시간 = {t_cpu/5}(s)')
        print(f'gpu 계산시간 = {t_gpu/5}(s)')
        print(f'GPU 성능은 CPU의 {t_cpu/t_gpu}배')


def plot_sparse(file):

    content = loadmat(file)
    a = content['Problem'][0, 0]['A']

    n = a.shape[0]

    if ssp.issparse(a):
        plt.figure(figsize=(5,5))
        plt.spy(a, color = 'k', markersize=0.5)
        plt.title(f'Matrix pattern ({n}x{n}, non-zeros: {a.nnz})')
        plt.xlabel('column')
        plt.ylabel('row')

    else:
        print("로드된 변수 'A'는 희소 행렬 형식이 아닙니다.")
        return
