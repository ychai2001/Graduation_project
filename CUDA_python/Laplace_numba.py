from matplotlib import pyplot as plt

import numpy as np
import math
from numpy import linalg
from numba import cuda
import time

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

n= 124
tol = 1e-4
xi = np.linspace(0, 1, n+2)
xx, yy = np.meshgrid(xi[1:-1], xi[1:-1])

@cuda.jit
def jacobi_gpu(n, ti, dt):
    i, j = cuda.grid(2)
    if 1 <= i <= n and 1 <= j <= n:
        dt[i, j] = 0.25*(ti[i-1, j] + ti[i, j-1] + ti[i+1, j] + ti[i, j+1]) - ti[i, j]

# 1~n까지가 내부 점, 경게조건은 외부
@cuda.jit
def bc_gpu(t):
    n = t.shape[0] - 2  # 내부 셀 개수
    i = cuda.grid(1)

    if 1 <= i <= n:
        t[0, i] = 100        # 아래쪽
        t[n+1, i] = 300      # 위쪽
        t[i, 0] = 100        # 왼쪽
        t[i, n+1] = 100      # 오른쪽

@cuda.jit
def add_array_gpu(ti, dt):
    i, j = cuda.grid(2)
    if i < ti.shape[0] and j < ti.shape[1]:
        ti[i, j] += dt[i, j]

def Laplace_gpu(n, tol):

    ti = np.zeros((n+2, n+2))
    dt = np.zeros((n+2, n+2))
    
    d_ti = cuda.to_device(ti)
    d_dt = cuda.to_device(dt)

    # 2차원 그리드 결정
    threadsperblock = (16,16)
    blockspergrid_x = math.ceil((n+2) / threadsperblock[0])
    blockspergrid_y = math.ceil((n+2) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    err = 1
    hist_jacobi = []

    while err > tol:
        
        # 경계 조건 일단 1D n개 쓰레드
        bc_gpu[1,n+2](d_ti)


        jacobi_gpu[blockspergrid, threadsperblock](n, d_ti, d_dt)
        #cuda.synchronize()
 
        dt = d_dt.copy_to_host()
        err = linalg.norm(dt) / n
        hist_jacobi.append(err)

        add_array_gpu[blockspergrid, threadsperblock](d_ti, d_dt)
        #cuda.synchronize()

    ti = d_ti.copy_to_host()
    
    return ti, hist_jacobi


ns = np.arange(30,127,8)
times_numba =[]

for n in ns:
    t_s_gpu = time.time()
    Laplace_gpu(n, tol)
    t_e_gpu = time.time()
    times_numba.append(t_e_gpu - t_s_gpu)

np.save("time_numba.npy", times_numba)
