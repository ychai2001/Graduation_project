from matplotlib import pyplot as plt

import numpy as np
import math
from numpy import linalg
from numba import cuda
import time

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

@cuda.jit
def jacobi_gpu(n, ti, dt):
    i, j, k = cuda.grid(3)
    if 1 <= i <= n and 1 <= j <= n and 1 <= k <= n:
        dt[i, j, k] = 1/6 * (
            ti[i-1, j, k] + ti[i+1, j, k] +
            ti[i, j-1, k] + ti[i, j+1, k] +
            ti[i, j, k-1] + ti[i, j, k+1]
        ) - ti[i, j, k]

# 1~n까지가 내부 점, 경게조건은 외부
@cuda.jit
def bc_gpu(t):
    n = t.shape[0] - 2  # 내부 셀 개수
    i, j = cuda.grid(2)

    if 1 <= i <= n and 1 <= j <= n:
        t[i, n+1, j] = 300   # upper side
        t[i, 0, j] = 100    # lower side
        t[i, j, n+1] = 100   # Right
        t[i, j, 0] = 100    # Left
        t[0, i, j] = 100    # Front
        t[n+1, i, j] = 100   # Back


@cuda.jit
def add_array_gpu(ti, dt):
    i, j, k = cuda.grid(3)
    if i < ti.shape[0] and j < ti.shape[1] and k < ti.shape[2]:
        ti[i, j, k] += dt[i, j, k]

@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def square_gpu(matrix, result):
    i, j, k = cuda.grid(3)
    if i < matrix.shape[0] and j < matrix.shape[1] and k < matrix.shape[2]:
        result[i, j, k] = matrix[i, j, k] * matrix[i, j, k]

def Laplace_gpu(n, tol):

    ti = np.zeros((n+2, n+2, n+2))
    dt = np.zeros((n+2, n+2, n+2))
    
    d_ti = cuda.to_device(ti)
    d_dt = cuda.to_device(dt)
    d_dt_squared = cuda.device_array_like(d_dt)
    
    # 3차원 Thread Layout 결정
    threadsperblock = (8,8,8)
    blockspergrid_x = math.ceil((n+2) / threadsperblock[0])
    blockspergrid_y = math.ceil((n+2) / threadsperblock[1])
    blockspergrid_z = math.ceil((n+2) / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # bc위한 2D Layout
    tpb = (16,16)
    bpg_x = math.ceil((n+2) / tpb[0])
    bpg_y = math.ceil((n+2) / tpb[1])
    bpg = (bpg_x, bpg_y)

    err = 1
    hist_jacobi = []

    while err > tol:
        
        # 경계 조건 2D n개 쓰레드
        bc_gpu[bpg, tpb](d_ti)

        jacobi_gpu[blockspergrid, threadsperblock](n, d_ti, d_dt)

        square_gpu[blockspergrid, threadsperblock](d_dt, d_dt_squared)
        cuda.synchronize()
        err = math.sqrt(sum_reduce(d_dt_squared.ravel())) / n
        hist_jacobi.append(err)

        add_array_gpu[blockspergrid, threadsperblock](d_ti, d_dt)

    ti = d_ti.copy_to_host()
    
    return ti, hist_jacobi

n= 62
tol = 1e-4
xi = np.linspace(0, 1, n+2)
ti_gpu, _ = Laplace_gpu(n, tol)
np.save("ti_numba.npy", ti_gpu)

# for y in range(1, 32, 2):
#     plt.imshow(ti_gpu[:, y, :])
#     plt.title(f'Slice at y={y}')
#     plt.colorbar()
#     plt.show()


ns = np.arange(30,63,8)
tol = 1e-4
times_numba =[]

for n in ns:
    t_s_gpu = time.time()
    Laplace_gpu(n, tol)
    t_e_gpu = time.time()
    times_numba.append(t_e_gpu - t_s_gpu)

np.save("time_numba.npy", times_numba)
