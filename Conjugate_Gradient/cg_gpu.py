import numpy as np
from numba import cuda
import gpu_func as gpu


def cg_gpu(A_h, b_h, x0_h, N, max_iter=5000, tol=1e-12):
    # Host 데이터를 Device로 전송
    A_d = cuda.to_device(A_h)
    b_d = cuda.to_device(b_h)
    x_d = cuda.to_device(x0_h)
    
    # GPU 임시/결과 벡터 할당
    r_d = cuda.device_array(N, dtype=np.float32)
    p_d = cuda.device_array(N, dtype=np.float32)
    Ap_d = cuda.device_array(N, dtype=np.float32)
    tmp_d = cuda.device_array(N, dtype=np.float32) # 내적 계산용

    # 1D 벡터 연산을 위한 스레드/블록 설정
    threadsperblock = 256
    blockspergrid = (N + threadsperblock - 1) // threadsperblock

    # 1. 초기화: r0 = b - A*x0, p0 = r0
    gpu.matvecdot(A_d, x_d, Ap_d)
    gpu.vecsub[blockspergrid, threadsperblock](b_d, Ap_d, r_d)
    p_d = cuda.device_array_like(r_d)
    gpu.vec_copy[blockspergrid, threadsperblock](r_d, p_d)

    rho_prev = gpu.vecdot(r_d, r_d, tmp_d)

    for j in range(max_iter):
        # 2. A*p_k 계산 (Ap_d = A * p_d) -> 병렬 mat_vec_mult
        gpu.matvecdot(A_d, p_d, Ap_d)

        # 3. alpha_k 계산: alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        pAp = gpu.vecdot(p_d, Ap_d, tmp_d)
        if pAp == 0.0: break
        alpha = rho_prev / pAp

        # 4. x_{k+1} = x_k + alpha_k * p_k -> 병렬 svmul 및 vec_add
        p_d_scaled = cuda.device_array(N, dtype=np.float32)
        gpu.svmul[blockspergrid, threadsperblock](p_d, alpha, p_d_scaled)
        gpu.vecadd[blockspergrid, threadsperblock](x_d, p_d_scaled, x_d) # x_d in-place 업데이트
        
        # 5. r_{k+1} = r_k - alpha_k * A * p_k -> 병렬 svmul 및 vec_sub
        Ap_d_scaled = cuda.device_array(N, dtype=np.float32)
        gpu.svmul[blockspergrid, threadsperblock](Ap_d, alpha, Ap_d_scaled)
        r_new_d = cuda.device_array(N, dtype=np.float32)
        gpu.vecsub[blockspergrid, threadsperblock](r_d, Ap_d_scaled, r_new_d)
        
        # 6. 수렴 확인 및 rho_next 계산
        rho_next = gpu.vecdot(r_new_d, r_new_d, tmp_d)
        if np.sqrt(rho_next) < tol:
            return x_d.copy_to_host(), j

        # 7. beta_k 계산: beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        beta = rho_next / rho_prev
        
        # 8. p_{k+1} = r_{k+1} + beta_k * p_k -> 병렬 svmul 및 vec_add
        gpu.svmul[blockspergrid, threadsperblock](p_d, beta, p_d_scaled) # p_k에 beta_k 곱
        p_new_d = cuda.device_array(N, dtype=np.float32)
        gpu.vecadd[blockspergrid, threadsperblock](r_new_d, p_d_scaled, p_new_d)

        # 9. 다음 반복을 위한 값 업데이트
        rho_prev = rho_next
        r_d = r_new_d
        p_d = p_new_d

    # 최대 반복 횟수 초과 시
    return x_d.copy_to_host(), max_iter