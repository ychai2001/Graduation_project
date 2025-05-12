from matplotlib import pyplot as plt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
from numpy import linalg

import math
import time

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

# Generate points (excluding BC)
n= 124
tol = 1e-4
xi = np.linspace(0, 1, n+2)
xx, yy = np.meshgrid(xi[1:-1], xi[1:-1])

# PyCUDA 커널 정의
module = SourceModule("""
__global__ void bc_gpu(int n, float* t) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= 1 && i <= n) {
        int N = n + 2;

        // 아래쪽 (row 0)
        t[i] = 100;

        // 위쪽 (row n+1)
        t[(n+1)*N + i] = 300;

        // 왼쪽 (col 0)
        t[i * N] = 100;

        // 오른쪽 (col n+1)
        t[i * N + (n+1)] = 100;
    }
}

__global__ void jacobi_gpu(int n, float* ti, float* dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = n + 2;

    if (i >= 1 && i <= n && j >= 1 && j <= n) {
        int idx = i + j * N;
        dt[idx] = 0.25f * (ti[idx - 1] + ti[idx + 1] + ti[idx - N] + ti[idx + N]) - ti[idx];
    }
}

__global__ void add_array_gpu(int n, float* ti, float* dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = n + 2;

    if (i < N && j < N) {
        int idx = i + j * N;
        ti[idx] += dt[idx];
    }
}
""")

# 함수 로딩
bc_gpu = module.get_function("bc_gpu")
jacobi_gpu = module.get_function("jacobi_gpu")
add_array_gpu = module.get_function("add_array_gpu")

# Jacobi solver 함수
def Laplace_pcd(n, tol):
    
    ti = np.zeros((n+2, n+2), dtype=np.float32)
    dt = np.zeros_like(ti)

    # GPU 메모리 할당
    d_ti = cuda.mem_alloc(ti.nbytes)
    d_dt = cuda.mem_alloc(dt.nbytes)

    # 초기값 복사
    cuda.memcpy_htod(d_ti, ti)
    cuda.memcpy_htod(d_dt, dt)

    threadsperblock = (16, 16, 1)
    blockspergrid = (math.ceil((n+2) / 16), math.ceil((n+2) / 16), 1)
    block1d = (64, 1, 1)
    grid1d = ((n + block1d[0] - 1) // block1d[0], 1, 1)

    err = 1
    hist_jacobi = []

    while err > tol:
        # 경계 조건 적용
        bc_gpu(np.int32(n), d_ti, block=block1d, grid=grid1d) # 그동안 다르게 n값만 받아감, 알아서 위에 함수에서 처리

        # Jacobi 연산
        jacobi_gpu(np.int32(n), d_ti, d_dt, block=threadsperblock, grid=blockspergrid) # 넘바와 동일

        # 결과 복사 및 오차 계산
        cuda.memcpy_dtoh(dt, d_dt)  # 호스트로 보내서 놈 계산
        err = linalg.norm(dt[1:-1, 1:-1]) / n
        hist_jacobi.append(err)

        # ti += dt
        add_array_gpu(np.int32(n), d_ti, d_dt, block=threadsperblock, grid=blockspergrid)   # 업데이트

    # 결과 복사
    cuda.memcpy_dtoh(ti, d_ti)  #호스트로 복사
    return ti, hist_jacobi

# ti_pcd, _ = Laplace_pcd(n, tol)

# # Plot contour
# plt.contourf(xx, yy, ti_pcd[1:-1, 1:-1], levels=10)
# plt.title('Laplace Solution via GPU Computing')
# plt.colorbar()
# plt.show()

# ns = np.arange(30,127,8).astype(int)
ns = [30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126]
times_pycuda =[]

for n in ns:
    t_s_gpu = time.time()
    Laplace_pcd(n, tol)
    t_e_gpu = time.time()
    times_pycuda.append(t_e_gpu - t_s_gpu)

print(len(times_pycuda))
np.save("time_pycuda.npy", times_pycuda)

