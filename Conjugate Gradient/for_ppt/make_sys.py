from scipy.io import loadmat
import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np
import cupy as cp
import time
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg

from matplotlib import pyplot as plt

def make_A(n, W, diag=1.0):
    """
    주어진 크기(n)와 대역폭(W)을 갖는 symetric, positive definite, sparse matrix 생성

    parameters
    ---------------------------
    n: 행렬의 크기 (n x n)
    W: 대역폭  1 <= W < n,  W=1은 3중 대각 행렬을 의미.
    diag: 대각 우위의 정도. 값이 클수록 대각 성분이 지배적

    return
    ---------------------------
    A : CSR 형식의 symetric, positive definite, sparse matrix (dtype=float32)
    """
    if W < 1 or W >= n:
        raise ValueError("bandwidth 범위 오류")

    # 1. 0이 아닌 성분들의 인덱스 생성
    # 대역폭 W 이내의 인덱스만 고려하여 i, j 배열생성
    rows = []
    cols = []
    data = []
    
    # Upper Triangle 부분만 생성하여 대칭성 보장
    for i in range(n):
        for j in range(i, min(n, i + W)): # 주대각선(j=i)부터 대역폭 W까지
            
            # 2. 비대각 성분의 값 설정 (a_ij, i != j)
            if i != j:
                # 무작위 값을 사용
                val = np.random.rand() * 0.5  
                
                rows.append(i)
                cols.append(j)
                data.append(val)
                
                # 대칭 성분 추가 (a_ji = a_ij)
                rows.append(j)
                cols.append(i)
                data.append(val)
    
    # 3. 대각 성분의 값 설정 (a_ii)
    # 각 행의 비대각 성분 합보다 큰 값을 부여하여 '대각 우위'를 만듭니다.
    # 이는 행렬의 양의 정부호성을 보장하는 일반적인 방법입니다.
    
    # 비대각 성분의 절댓값 합을 계산 (대략적인 값)
    off_diag_sum = np.zeros(n, dtype=np.float32)
    for r, c, val in zip(rows, cols, data):
        if r != c:
            off_diag_sum[r] += abs(val)

    # 대각 성분 생성 (a_ii > sum(|a_ij|))
    for i in range(n):
        diag_val = off_diag_sum[i] * diag + 1.0 + np.random.rand() * 5.0
        
        rows.append(i)
        cols.append(i)
        data.append(diag_val)
    
    # 4. 희소 행렬 생성 (COO -> CSR 형식)
    A = ssp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    
    # 5. 생성된 행렬의 정보 출력
    #print(f"생성된 행렬의 크기: {A.shape}")
    #print(f"0이 아닌 성분 개수 (nnz): {A.nnz}\n")
    
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

def cal_time(a):
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
    
    b, x = make_b(a)

    a = ssp.csr_matrix(a, dtype=np.float32)
    A = csp.csr_matrix(a, dtype=np.float32)
    B = cp.asarray(b, dtype=np.float32)   # 이거는 cpu to gpu같은거

    t_cpu= 0
    t_gpu = 0

    for i in range(5):
        ts_cpu = time.time()
        sol_cpu, r_cpu = ssp_linalg.cg(a, b)
        te_cpu = time.time()
        t_cpu += te_cpu-ts_cpu

    for j in range(5):
        ts_gpu = time.time()
        sol_gpu, r_gpu = csp_linalg.cg(A, B)
        te_gpu = time.time()
        t_gpu += te_gpu-ts_gpu
    sol_gpu = sol_gpu.get()

    # scipy와 cupyx.scipy에서 계산 후 반환값이 0 이면 max iter도달 전 수렴했다는 의미
    if r_cpu == 0 and r_gpu == 0:
        diff = np.linalg.norm(sol_cpu - sol_gpu)
        
        if np.allclose(sol_cpu, sol_gpu, atol=1e-5):
            #print(f'cpu 계산시간 = {t_cpu/5}(s)')
            #print(f'gpu 계산시간 = {t_gpu/5}(s)')
            #print(f'GPU 성능은 CPU의 {t_cpu/t_gpu}배\n')
            return t_cpu/5, t_gpu/5
        else:
            print(f'Exact solution = {x}\n')
            print(f'CPU solution L2 Error = {np.linalg.norm(sol_cpu - x)}')
            print(f'GPU solution L2 Error = {np.linalg.norm(sol_gpu - x)}\n')
            raise ValueError(f"Solution doesn't match : {diff}")
            
    else:
        raise ValueError('Solution not converged')


def plot_sparse(a):
    """
    sparse matrix A의 원소 분포를 plot하는 함수, 값이 존재하면 검은색 아니면 흰색

    parameters
    ---------------------
    a : sparse, symetric, positive definite matrix from function 'make_A'

    """

    n = a.shape[0]

    if ssp.issparse(a):
        plt.figure(figsize=(8,8))
        plt.spy(a, color = 'k', markersize=0.1)
        plt.title(f'Matrix pattern ({n}x{n}, non-zeros: {a.nnz})')
        plt.xlabel('column')
        plt.ylabel('row')

    else:
        print("A is not a sparse matrix")
        return
    
def cond(A):
    """
    A행렬의 조건수 계산

    parameters
    ----------------------
    A : sparse, symetric matrix (coo or csr)

    """

    if not ssp.isspmatrix_csr(A):
        # COO 또는 다른 형식이면 CSR로 변환
        A_cal = A.tocsr()
    else:
        A_cal = A
    # 최대 고유값
    lambda_max = ssp_linalg.eigsh(A_cal, k=1, which='LM', return_eigenvectors=False)

    # 최소 고유값
    lambda_min = ssp_linalg.eigsh(A_cal, k=1, which='SM', return_eigenvectors=False)

    # 조건수 계산
    if abs(lambda_min[0]) > 1e-12:
        cond_A = abs(lambda_max[0] / lambda_min[0])
        #print(f"최대 고유값 : {lambda_max[0]:.4f}")
        #print(f"최소 고유값 : {lambda_min[0]:.4f}\n")
        #print(f"행렬 A의 조건수: {cond_A:.4f}")
    else:
        print("가장 작은 고유값이 0에 매우 가까워 조건수가 무한대에 가깝습니다.")
    return cond_A