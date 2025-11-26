import scipy.sparse as ssp
import scipy.sparse.linalg as ssp_linalg
import numpy as np

def cond(A):
    """
    A행렬의 조건수 계산

    parameters
    ----------------------
    A : sparse, symetric matrix (coo or csr)

    """
    if not ssp.issparse(A):
        raise TypeError("Input should be sparse matrix")

    if not ssp.isspmatrix_csr(A):
        # COO 또는 다른 형식이면 CSR로 변환
        A_cal = A.tocsr(copy = True)
    else:
        A_cal = A.copy()

    # 최대 고유값
    lambda_max = ssp_linalg.eigsh(A_cal, k=1, which='LM', return_eigenvectors=False)
    print(f"최대 고유값 : {lambda_max[0]}")
    # 최소 고유값
    lambda_min = ssp_linalg.eigsh(A_cal, k=1, which='LM', sigma = 0, return_eigenvectors=False)
    print(f"최소 고유값 : {lambda_min[0]}")

    # 조건수 계산
    if abs(lambda_min) > 1e-12:
        cond_A = abs(lambda_max[0] / lambda_min[0])
        #print(f"최대 고유값 : {lambda_max[0]:.4f}")
        #print(f"최소 고유값 : {lambda_min[0]:.4f}\n")
    else:
        print("가장 작은 고유값이 0에 매우 가까워 조건수가 무한대에 가깝습니다.")

    return cond_A


def diag_ratio(matrix):
    if not ssp.issparse(matrix):
        raise TypeError("Input should be sparse matrix")

    M, N = matrix.shape
    if M != N:
        raise ValueError("정방 행렬(M=N)만 지원합니다.")

    if not ssp.isspmatrix_csr(matrix):
        # COO 또는 다른 형식이면 CSR로 변환
        A = matrix.tocsr(copy = True)
    else:
        A = matrix.copy()
    
    A.data = np.abs(A.data)

    abs_row_sums = A.sum(axis=1).A1
    abs_diagonals = A.diagonal()
    off_diagonal_sums = abs_row_sums - abs_diagonals
    
    diag = np.divide(
        abs_diagonals, 
        off_diagonal_sums, 
        out=np.zeros_like(abs_diagonals, dtype=float), 
        where=off_diagonal_sums != 0
    )
    
    return diag

def bandwidth(matrix):
    """
    희소 행렬의 대역폭 (Bandwidth = bl + bu + 1)을 계산합니다.

    Args:
        matrix: 계산할 scipy.sparse 행렬.

    Returns:
        int: 행렬의 대역폭.

    Raises:
        TypeError: 입력이 scipy.sparse 행렬이 아닐 때 발생합니다.
    """
    if not ssp.issparse(matrix):
        raise TypeError("Input should be sparse matrix")
    
    if not ssp.isspmatrix_coo(matrix):
        # 다른 형식이면 COO로 변환
        A = matrix.tocoo(copy = True)
    else:
        A = matrix.copy()

    if A.nnz == 0:
        return 0
    
    # 0이 아닌 요소의 행 인덱스 (i)와 열 인덱스 (j) 추출
    rows = A.row
    cols = A.col
    
    # 1. 상부 대역폭 (Upper Bandwidth, bu): max(j - i)
    # 주 대각선 위쪽으로 가장 멀리 떨어진 0이 아닌 요소의 거리
    upper_band = np.max(cols - rows)
    
    # 2. 하부 대역폭 (Lower Bandwidth, bl): max(i - j)
    # 주 대각선 아래쪽으로 가장 멀리 떨어진 0이 아닌 요소의 거리
    lower_band = np.max(rows - cols)
    
    # 3. 전체 대역폭 계산: Bandwidth = bl + bu + 1
    # 참고: +1은 주 대각선 자체(j-i=0)를 포함하기 위함입니다.
    bandwidth = lower_band + upper_band + 1
    
    return int(bandwidth)

def size(matrix):
    if not ssp.issparse(matrix):
        raise TypeError("Input should be sparse matrix")
    m, n = matrix.shape
    size = m*n
    return size

def character(a):
    c = cond(a)
    d = np.mean(diag_ratio(a))
    b = bandwidth(a)
    n = size(a)
    r_nnz = a.nnz / n
    print(f'Condition Number : {c}')
    print(f'Average Diagonlity : {d}')
    print(f'Bandwidth : {b}')
    print(f'Bandwidth Ratio : {b/np.sqrt(n)}')
    print(f'Non-zeros Ratio : {r_nnz*100} % \n')
