import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np
import cupy as cp
import time
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg
import gc
from cg_solver import cg_cp, cg_cpu

from matplotlib import pyplot as plt

def make_A(n, W, r_nnz, diag=2.0):
    """
    주어진 크기(n), 대역폭(W), 전체 행렬 대비 non-zero 비율(r_nnz)을 갖는
    symetric, positive definite, sparse matrix 생성 (불가능한 비율 입력 시 오류 반환)

    parameters
    ---------------------------
    n: 행렬의 크기 (n x n)
    W: 대역폭 (1 <= W < n)
    r_nnz: 전체 행렬(n*n) 대비 0이 아닌 요소의 비율 (0.0 < r_nnz < 1.0)
    diag: 대각 우위의 정도.
    
    return
    ---------------------------
    A : CSR 형식의 symetric, positive definite, sparse matrix (dtype=float32)
    """
    if W < 1 or W >= n:
        raise ValueError("bandwidth 범위 오류: 1 <= W < n 이어야 합니다.")
    if not (0.0 < r_nnz < 1.0):
        raise ValueError("r_nnz 범위 오류: 0.0 < r_nnz < 1.0 이어야 합니다.")

    # W이내에 존재하는 최대 원소 갯수 (upper tringle)
    if W<= 1:
        nnz_max_band_upper = 0
    else:
        nnz_max_band_upper = (n - W + 1)*(W - 1) + ((W - 2)*(W - 1)) // 2
    
    # r_nnz를 기반으로 주 대각선(n개)을 제외한 필요한 비대각 성분 개수(upper tri)
    num_non_diag_nnz_upper = int(np.ceil(int(np.ceil(n * n * r_nnz)) - n) / 2)
    
    # 가능한 최대 nnz보다 r_ratio로 계산된 양이 많을 때
    if num_non_diag_nnz_upper > nnz_max_band_upper:
        max_possible_ratio = (2 * nnz_max_band_upper + n) / (n * n)
        
        raise ValueError(
            f"r_nnz({r_nnz:.4f}) passes a limit of Bandwidth({W})\n"
            f"Maximum possible r_nnz is about {max_possible_ratio:.4f}\n"
        )
    
    # 3. 비대각 성분을 채울 전체 후보 인덱스 쌍 생성 (상삼각 영역, 대역폭 W 내)
    all_upper_indices = []
    for i in range(n):
        # 주대각선(i)을 제외하고, 대역폭 W 이내의 인덱스 (i+1 ~ i+W-1)
        for j in range(i + 1, min(n, i + W)): 
            all_upper_indices.append((i, j))
    
    # 4. 후보 인덱스에서 무작위로 'num_non_diag_nnz_upper' 개만큼 선택
    if all_upper_indices:
        np.random.seed(0) # 재현성을 위해 시드 설정 (선택 사항)
        sampled_indices_idx = np.random.choice(
            len(all_upper_indices), 
            size=num_non_diag_nnz_upper, 
            replace=False
        )
        selected_upper_indices = [all_upper_indices[idx] for idx in sampled_indices_idx]
    else:
        selected_upper_indices = []

    # 5. 행렬 데이터 생성 (비대각 성분)
    rows = []
    cols = []
    data = []

    for i, j in selected_upper_indices:
        val = np.random.rand()  # 무작위 값 (0 ~ 1)
        
        # 상삼각 성분 (a_ij) 및 대칭 성분 (a_ji = a_ij) 추가
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([val, val])
        
    # 6. 대각 성분의 값 설정 (a_ii)
    # 비대각 성분만으로 임시 희소 행렬 생성 (CSR 변환이 효율적)
    if rows:
        A_off = ssp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        # 각 행의 비대각 성분 합
        off_diag_sum = A_off.sum(axis=1).A1
    else:
        off_diag_sum = np.zeros(n, dtype=np.float32)

    # 대각 성분 생성 (대각 우위 보장: a_ii > sum(|a_ij|))
    for i in range(n):
        # diag 인수를 곱하여 대각 성분 값을 결정
        diag_val = off_diag_sum[i] * diag
        
        rows.append(i)
        cols.append(i)
        data.append(diag_val)
    
    # 7. 최종 희소 행렬 생성
    A = ssp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    
    return A


def make_b(A, x_true=None):
    """
    계산이 가능하도록 x_true로부터 b벡터를 구성
    

    parameters
    ------------------------------------
    A : symetric, positive definite matrix
    x_ture : Exact solution or random vector with length of row or column of A matrix


    return
    ------------------------------------
    b : RHS of system Ax=b with parameter A and solution x_true
    x_true : return for check error from computing
    """
    n = A.shape[0]

    if x_true == None:
        x_true = np.random.rand(n).astype(np.float32)
        b = A.dot(x_true).astype(np.float32)
    else:
        b = A.dot(x_true).astype(np.float32)

    return b, x_true

def cal_time(a, tol=1e-5, residual=True):
    """
    A 행렬을 받아와 conjugate gradient method로 Ax=b 계산 시간 비교
    b, x는 make_b 함수로부터 계산
    데이터 타입은 모두 float32로 통일

    parameters
    -------------------------
    a : sparse, symetric, positive definite matrix


    return
    -------------------------
    t_cpu : Computing time for Ax=b in CPU
    t_gpu : Computing time for Ax=b in GPU
    """
    # allclose의 tol설정
    b, x = make_b(a)
    bnorm = np.linalg.norm(b)
    tol = tol*bnorm

    a = ssp.csr_matrix(a, dtype=np.float32)
    A = csp.csr_matrix(a, dtype=cp.float32)
    B = cp.asarray(b, dtype=cp.float32)   # 이거는 cpu to gpu같은거

    residuals_cpu = []
    residuals_gpu = []
    residuals_nb = []
    residuals_cupy = []

    def get_res(xk):
        r_k = b - a.dot(xk)
        residuals_cpu.append(np.linalg.norm(r_k))

    def get_res_nb(xk):
        r_k = b - a.dot(xk)
        residuals_nb.append(np.linalg.norm(r_k))

    def get_res_gpu(xk):
        r_k = B - A.dot(xk)
        residuals_gpu.append(cp.linalg.norm(r_k).get())

    def get_res_cupy(xk):
        r_k = B - A.dot(xk)
        residuals_cupy.append(cp.linalg.norm(r_k).get()) 

    t_cpu= 0
    t_gpu = 0
    t_nb = 0
    t_cupy = 0

    mempool = cp.get_default_memory_pool()

    #scipy sparse cpu시간
    ssp_linalg.cg(a, b, callback=get_res if residual else None) #한번 계산 버릴 겸 residual 계산
    for i in range(20):
        ts_cpu = time.time()
        sol_cpu, r_cpu = ssp_linalg.cg(a, b)
        te_cpu = time.time()
        if i >= 10:
            t_cpu += te_cpu-ts_cpu

    # cpu병렬화 버전
    cg_cpu(a, b, callback=get_res_nb if residual else None) #한번 계산 버릴 겸 residual 계산
    for j in range(20):
        ts_nb = time.time()
        sol_nb, it_nb = cg_cpu(a, b)
        te_nb = time.time()
        if j >= 10:
            t_nb += te_nb-ts_nb

    #cupy sparse gpu 시간
    csp_linalg.cg(A, B, callback=get_res_gpu if residual else None) #한번 계산 버릴 겸 residual 계산
    for k in range(20):
        ts_gpu = time.time()
        sol_gpu, r_gpu = csp_linalg.cg(A, B)
        te_gpu = time.time()
        if k >= 10:
            t_gpu += te_gpu-ts_gpu
    sol_gpu = sol_gpu.get()
    gc.collect()
    mempool.free_all_blocks()

    # gpu 병렬화 버전
    cg_cp(A, B, callback=get_res_cupy if residual else None) #한번 계산 버릴 겸 residual 계산
    for l in range(20):
        ts_cupy = time.time()
        sol_cupy, it_cupy = cg_cp(A, B)
        te_cupy = time.time()
        if l >= 10:
            t_cupy += te_cupy-ts_cupy
    sol_cupy = sol_cupy.get()
    gc.collect()
    mempool.free_all_blocks()

    # scipy와 cupyx.scipy에서 계산 후 반환값이 0 이면 max iter도달 전 수렴했다는 의미
    if r_cpu == 0 and r_gpu == 0:
        
        if np.allclose(sol_cpu, x, atol= tol) and np.allclose(sol_gpu, x, atol=tol) and np.allclose(sol_nb, x, atol=tol) and np.allclose(sol_cupy, x, atol=tol):
            if residual:
                return t_cpu/5, t_gpu/5, t_nb/5, t_cupy/5, residuals_cpu, residuals_gpu, residuals_nb, residuals_cupy
            else:
                return t_cpu/5, t_gpu/10, t_nb/5, t_cupy/5
        
        else:
            print(f'Exact solution = {x}\n')
            print(f'CPU solution L2 Error = {np.linalg.norm(sol_cpu - x)}')
            print(f'GPU solution L2 Error = {np.linalg.norm(sol_gpu - x)}\n')
            print(f'cpu 계산시간 : {t_cpu/5}')
            print(f'gpu 계산시간 : {t_gpu/5}')
            raise ValueError(f"Solution doesn't match, tol : {tol}, cpu :{len(residuals_cpu)+1}회 계산, gpu :{len(residuals_gpu)+1}회 계산")
            
    else:
        raise ValueError('Solution not converged')

def plot_sparse(a, color=False):
    """
    sparse matrix A의 원소 분포를 plot하는 함수

    parameters
    ---------------------
    a : sparse, symetric, positive definite matrix from function 'make_A'

    """

    n = a.shape[0]

    if not ssp.issparse(a):
        print("A is not a sparse matrix")
        return
    
    if color == False:
        plt.figure(figsize=(8, 8))
        plt.spy(a, color = 'k', markersize=0.1)
        plt.title(f'Matrix pattern ({n}x{n}, non-zeros: {a.nnz})')
        plt.xlabel('column')
        plt.ylabel('row')
        plt.show()
    else:
        a_coo = a.tocoo()
    
        rows = a_coo.row
        cols = a_coo.col
        data = a_coo.data # 0이 아닌 값들

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            cols, 
            rows, 
            c=data, 
            cmap='RdBu_r', # 빨간색(양수)과 파란색(음수)을 잘 나타내는 컬러맵 (RdBu_r 권장)
            s=1
        )
        plt.gca().invert_yaxis()
        plt.xlim(-0.5, n - 0.5)
        plt.ylim(n - 0.5, -0.5)
        plt.colorbar(label='Element Value in Matrix A')
        plt.title(f'Matrix A Value and Pattern ({n}x{n}, NNZ: {a.nnz})')
        plt.grid(True, alpha=0.7)
        plt.show()

    





