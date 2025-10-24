from numba import cuda

@cuda.jit
def matmul(A, B, result):
    """
    Perform square matrix multiplication of result = A * B
    """
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        result[i, j] = tmp

@cuda.jit
def svmul(vec_in, scalar, result):
    """
    scalar - vector product
    """
    i = cuda.grid(1)

    if i < result.size:
        result[i] = vec_in[i] * scalar

@cuda.jit
def vecmul(V1, V2, result):
    """
    Not a dot product 그저 요소끼리의 곱을 저장하는 함수임
    """
    i = cuda.grid(1)
    if i < result.size:
        result[i] = V1[i] * V2[i]

@cuda.jit
def vecadd(V1, V2, result):
    i = cuda.grid(1)
    if i < result.size:
        result[i] = V1[i] + V2[i]

@cuda.jit
def vecsub(V1, V2, result):
    i = cuda.grid(1)
    if i < result.size:
        result[i] = V1[i] - V2[i]

@cuda.jit
def vec_copy(V_in, V_out):
    i = cuda.grid(1)
    if i < V_out.size:
        V_out[i] = V_in[i]

@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def square(matrix, result):
    i, j= cuda.grid(2)
    if i < matrix.shape[0] and j < matrix.shape[1]:
        result[i, j] = matrix[i, j ] * matrix[i, j]

def vecdot(V1_d, V2_d, tmp_d):
    """
    tmp_d는 사실상 안쓰이지만 반복적으로 메모리 할당, 해제보단 외부에서 더미 메모리로 설정하는게 나을듯
    그런 의미에서 N, tpb도 외부가 낫지만 차이가 미미할 것으로 예상됨
    """
    N = V1_d.size
    tpb = 256 # Threads per Block
    bpg = (N + tpb - 1) // tpb  # Blocks per Grid 계산

    vecmul[bpg, tpb](V1_d, V2_d, tmp_d)
    
    return sum_reduce(tmp_d)

def matvecdot(A_d, p_d, result_d):
    """GPU에서 행렬, 벡터 곱 A*p를 계산 (matmul 사용)"""
    N = p_d.shape[0]
    
    # 1D 벡터 p_d를 (N, 1) 행렬로 변환
    p_reshaped = p_d.reshape(N, 1)
    result_reshaped = result_d.reshape(N, 1)
    
    # 2D 그리드 설정
    threadsperblock_MV = (16, 1)
    blockspergrid_MV_x = (N + threadsperblock_MV[0] - 1) // threadsperblock_MV[0]
    blockspergrid_MV_y = (1 + threadsperblock_MV[1] - 1) // threadsperblock_MV[1]
    blockspergrid_MV = (blockspergrid_MV_x, blockspergrid_MV_y)

    matmul[blockspergrid_MV, threadsperblock_MV](A_d, p_reshaped, result_reshaped)