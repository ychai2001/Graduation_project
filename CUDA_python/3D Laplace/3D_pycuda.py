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

# PyCUDA 커널 정의
module = SourceModule("""
__global__ void bc_gpu(int n, float* t) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = n + 2;

    if (i >= 1 && i <= n && j >=1 && j <= n) {
        // ti[i, n+1, j] = 300   # top
        t[i * N * N + (n+1) * N + j] = 300.0f;

        // ti[i, 0, j] = 100     # bottom
        t[i * N * N + 0 * N + j] = 100.0f;

        // ti[i, j, n+1] = 100   # right
        t[i *N*N + j * N + (n+1)] = 100.0f;

        // ti[i, j, 0] = 100     # left
        t[i *N*N + j *N + 0] = 100.0f;

        // ti[0, i, j] = 100     # front
        t[0 *N*N + i * N + j] = 100.0f;

        // ti[n+1, i, j] = 100   # back
        t[(n+1)*N*N + i * N + j] = 100.0f;
    }
}

__global__ void jacobi_gpu(int n, float* ti, float* dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int nx = n + 2;
    int ny = n+2;

    if (i >= 1 && i <= n && j >= 1 && j <= n && k >= 1 && k <= n) {
        int idx = i*nx*ny + j*ny + k;
        dt[idx] = (1.0f / 6.0f) * (
            ti[(i-1) * nx * ny + j * ny + k] +
            ti[(i+1) * nx * ny + j * ny + k] +
            ti[i * nx * ny + (j-1) * ny + k] +
            ti[i * nx * ny + (j+1) * ny + k] +
            ti[i * nx * ny + j * ny + (k-1)] +
            ti[i * nx * ny + j * ny + (k+1)]
        ) - ti[idx];
    }
}

__global__ void add_array_gpu(int n, float* ti, float* dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int N = n + 2;

    if (i < N && j < N && k < N) {
        int idx = i*N*N + j*N + k;
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
    
    ti = np.zeros((n+2, n+2,n+2), dtype=np.float32)
    dt = np.zeros_like(ti)

    # GPU 메모리 할당
    d_ti = cuda.mem_alloc(ti.nbytes)
    d_dt = cuda.mem_alloc(dt.nbytes)

    # 초기값 복사
    cuda.memcpy_htod(d_ti, ti)
    cuda.memcpy_htod(d_dt, dt)

    threadsperblock = (8,8,8)
    blockspergrid_x = math.ceil((n+2) / threadsperblock[0])
    blockspergrid_y = math.ceil((n+2) / threadsperblock[1])
    blockspergrid_z = math.ceil((n+2) / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # bc위한 2D Layout
    tpb = (16,16,1)
    bpg_x = math.ceil((n+2) / tpb[0])
    bpg_y = math.ceil((n+2) / tpb[1])
    bpg = (bpg_x, bpg_y, 1)

    err = 1
    hist_jacobi = []

    while err > tol:
        # 경계 조건 적용
        bc_gpu(np.int32(n), d_ti, block=tpb, grid=bpg) # 그동안 다르게 n값만 받아감, 알아서 위에 함수에서 처리

        # Jacobi 연산
        jacobi_gpu(np.int32(n), d_ti, d_dt, block=threadsperblock, grid=blockspergrid) # 넘바와 동일

        # 결과 복사 및 오차 계산
        cuda.memcpy_dtoh(dt, d_dt)  # 호스트로 보내서 놈 계산
        err = linalg.norm(dt) / n
        hist_jacobi.append(err)

        # ti += dt
        add_array_gpu(np.int32(n), d_ti, d_dt, block=threadsperblock, grid=blockspergrid)   # 업데이트

    # 결과 복사
    cuda.memcpy_dtoh(ti, d_ti)  #호스트로 복사
    return ti, hist_jacobi


#Generate points (excluding BC)
n= 62
tol = 1e-4
xi = np.linspace(0, 1, n+2)
ti_gpu, _ = Laplace_pcd(n, tol)
np.save("ti_pycuda.npy", ti_gpu)


# for y in range(1, 32, 2):
#     plt.imshow(ti_gpu[:, y, :])
#     plt.title(f'Slice at y={y}')
#     plt.colorbar()
#     plt.show()

times_pycuda = np.empty(7)
for i in range(10):
    ns = [30, 38, 46, 54, 62, 70, 78]
    times =[]
    for n in ns:
        t_s_gpu = time.time()
        Laplace_pcd(n, tol)
        t_e_gpu = time.time()
        times.append(t_e_gpu - t_s_gpu)
    times_pycuda += times
np.save("time_pycuda.npy", times_pycuda/10)


