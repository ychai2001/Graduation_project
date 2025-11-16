from scipy.io import loadmat
import scipy.sparse as ssp

from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150


def plot_sparse(file):

    content = loadmat(file)
    # a = content['Problem'][0, 0]['A']
    a = content['A']

    n = a.shape[0]

    if ssp.issparse(a):
        plt.figure(figsize=(8,8))
        plt.spy(a, color = 'k', markersize=0.1)
        plt.title(f'Matrix pattern ({n}x{n}, non-zeros: {a.nnz})')
        plt.xlabel('column')
        plt.ylabel('row')

    else:
        print("로드된 변수 'A'는 희소 행렬 형식이 아닙니다.")
        return
    

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
    
def plot_laplace_with_value(a):
    """
    희소 행렬 A의 원소 분포와 값의 크기를 동시에 plot하는 함수.
    
    Parameters
    ---------------------
    a : sparse matrix (CSR, COO 등)
        라플라스 연산자 행렬 A
    """

    if not ssp.issparse(a):
        print("A is not a sparse matrix")
        return

    # 1. 시각화를 위해 COO(Coordinate) 형식으로 변환합니다.
    # COO 형식은 행(row), 열(col), 데이터(data)를 쉽게 추출할 수 있게 합니다.
    a_coo = a.tocoo()
    
    rows = a_coo.row
    cols = a_coo.col
    data = a_coo.data # 0이 아닌 값들

    n_rows = a.shape[0]
    n_nnz = a.nnz

    # 2. 값의 절대 크기를 시각화에 사용할 수 있도록 설정합니다.
    # 라플라스 행렬의 값은 주로 -4와 1로 구성되어 있습니다.
    # 값의 크기(절댓값)에 따라 점의 크기(s)를 다르게 할 수 있습니다.
    # 여기서는 값 자체(data)를 컬러맵에 연결하여 색상으로 값을 표현합니다.
    
    # 3. 플롯 생성
    plt.figure(figsize=(10, 8))
    
    # plt.scatter를 사용하여 각 점을 플롯합니다.
    # x: 열 인덱스 (cols), y: 행 인덱스 (rows)
    # c: 컬러맵 값 (data) -> 색상으로 값의 크기 표현
    # s: 마커 크기 (옵션) -> 모든 요소의 크기를 통일하거나 값에 따라 설정 가능
    scatter = plt.scatter(
        cols, 
        rows, 
        c=data, 
        cmap='RdBu_r', # 빨간색(양수)과 파란색(음수)을 잘 나타내는 컬러맵 (RdBu_r 권장)
        s=1,          # 마커 크기 (조절 가능)
        marker='s'     # 마커 모양 (점, 사각형 등)
    )
    
    # 행렬 형태와 동일하게 y축을 반전시킵니다.
    plt.gca().invert_yaxis()
    
    # 축 범위 설정
    plt.xlim(-0.5, n_rows - 0.5)
    plt.ylim(n_rows - 0.5, -0.5)
    
    # 컬러바 추가
    plt.colorbar(scatter, label='Element Value in Matrix A')
    
    plt.title(f'Matrix A Value and Pattern ({n_rows}x{n_rows}, NNZ: {n_nnz})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()