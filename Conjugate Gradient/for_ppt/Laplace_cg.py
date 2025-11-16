import numpy as np
import numba as nb
import cupy as cp

import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csp_linalg

import time

from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

@nb.jit(nopython=True)
def make_laplace_a(n):
    """
    Laplace operator matrix A using Numba for acceleration.
    
    Parameters
    ----------
    n : integer
        size of system (n*n grid points)
        
    Returns
    -------
    a : numpy.ndarray (n*n, n*n)
        Laplace operator matrix
    """
    # A 행렬의 크기는 (n*n, n*n)
    N = n * n
    a = np.zeros((N, N), dtype=np.float32)
    
    # i_row는 A 행렬의 행 인덱스, i_row = n*l + k 로 매핑됩니다.
    for i_row in range(N):
        # l과 k는 2D 격자 좌표 (l: 행, k: 열)
        l = i_row // n  # 2D 행 인덱스 (0 to n-1)
        k = i_row % n   # 2D 열 인덱스 (0 to n-1)
        
        # 주 대각선 요소: A[i_row, i_row]
        # 이는 -4*u(l,k) 항에 해당
        a[i_row, i_row] = -4.0
        
        # 이웃 요소: u(l+1, k), u(l-1, k), u(l, k+1), u(l, k-1)
        
        # 1. 오른쪽 이웃 (k+1)
        if k + 1 < n:
            j_col = n * l + (k + 1)
            a[i_row, j_col] = 1.0
            
        # 2. 왼쪽 이웃 (k-1)
        if k - 1 >= 0:
            j_col = n * l + (k - 1)
            a[i_row, j_col] = 1.0
            
        # 3. 위쪽 이웃 (l+1)
        if l + 1 < n:
            j_col = n * (l + 1) + k
            a[i_row, j_col] = 1.0
            
        # 4. 아래쪽 이웃 (l-1)
        if l - 1 >= 0:
            j_col = n * (l - 1) + k
            a[i_row, j_col] = 1.0
            
    return a

@nb.jit(nopython=True)
def make_laplace_b(n):
    """
    라플라스 방정식의 경계 조건을 포함하는 해 배열 b를 Numba로 가속하여 생성합니다.
    
    Parameters
    ----------
    n : integer
        격자점의 크기 (n*n)
        
    Returns
    -------
    b : numpy.ndarray (n*n)
        경계 조건 벡터
    """
    N = n * n
    # b 벡터를 0으로 초기화
    b = np.zeros(N, dtype=np.float32)
    
    # 1D 인덱스 i_row (0부터 N-1)를 순회합니다.
    for i_row in range(N):
        # 1D 인덱스를 2D 격자 좌표 (i: 행, j: 열)로 변환
        i = i_row // n  # 2D 행 인덱스 (0 to n-1)
        j = i_row % n   # 2D 열 인덱스 (0 to n-1)
        
        # 경계 조건 적용
        
        # Top (i == n-1)
        if i == n - 1:
            b[i_row] -= 300.0
            
        # Bottom (i == 0)
        if i == 0:
            b[i_row] -= 100.0
            
        # Right (j == n-1)
        if j == n - 1:
            b[i_row] -= 100.0
            
        # Left (j == 0)
        if j == 0:
            b[i_row] -= 100.0
            
    return b

def plot_laplace(a):
    """
    sparse matrix A의 원소 분포를 plot하는 함수, 값이 존재하면 검은색 아니면 흰색

    parameters
    ---------------------
    a : sparse matrix

    """

    n = a.shape[0]

    if ssp.issparse(a):
        plt.figure(figsize=(8,8))
        plt.spy(a, color = 'k', markersize=0.1)
        plt.title(f'Matrix pattern ({n}x{n}, non-zeros: {a.nnz})')
        plt.xlabel('column')
        plt.ylabel('row')
        plt.show()

    else:
        print("A is not a sparse matrix")
        return


def main_laplace(n=300, print=False):
    a = make_laplace_a(n)
    b = make_laplace_b(n)

    # CPU
    # Ax=b에서 A만 sparse로 바꾸면 되는듯?
    a = ssp.csr_matrix(a, dtype=np.float32)

    # GPU
    # sparse matrix를 바로 전달해주면 된다고 나와있음
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

    # Generate points (excluding BC)
    xi = np.linspace(0, 1, n+2)
    xx, yy = np.meshgrid(xi[1:-1], xi[1:-1])

    # Plot contour
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.contour(xx, yy, sol_cpu.reshape(n,n))
    plt.title('CPU')
    plt.colorbar()
    plt.subplot(122)
    plt.contour(xx, yy, sol_gpu.reshape(n,n))
    plt.title('GPU')
    plt.colorbar()
    plt.show()

    hardware = ['CPU', 'GPU']
    cal_time = [t_cpu/5, t_gpu/5]
    plt.title('Computing time(s) for Laplace Equation')
    bar1 = plt.bar(hardware, cal_time, width=0.3)
    for i, j in enumerate(bar1) :
        plt.text(i, j.get_height(), round(cal_time[i], 4), ha = 'center')
    plt.show()

    # 매트릭스 모양 플롯
    if print == True:
        plot_laplace(a)

    return cal_time