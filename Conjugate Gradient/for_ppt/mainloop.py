from make_sys import *
from sparse_character import cond

ns = [10000, 20000, 40000, 60000]
W_ratios = [0.01, 0.02, 0.04, 0.08]
nnzs = [0.001, 0.005, 0.01, 0.05]
diags = [1, 2, 4, 8]

file = 'test.csv'

with open(file, 'w') as f:
    # 1. 파일 헤더 (Header) 쓰기
    # n, W, nnz, d, t_cpu, t_gpu 순서로 저장
    header = "n, W_ratio, W_actual, nnz_ratio, diag, condition_number, t_cpu, t_gpu, speedup\n"
    f.write(header)

    for n in ns:
        for W_ratio in W_ratios:
            W = int(n*W_ratio)

            for nnz in nnzs:
                for d in diags:
                    try:
                        a = make_A(n, W, nnz, diag=d)
                        cn = cond(a)

                        t_cpu, t_gpu = cal_time(a)
                        speedup = t_cpu/t_gpu

                        data_row = f"{n}, {W_ratio}, {W}, {nnz}, {d}, {cn}, {t_cpu}, {t_gpu}, {speedup}\n"
                        f.write(data_row)
                        print(f"n={n}, W_ratio={W_ratio}, nnz={nnz}, diag={d}, speedup={speedup:.2f} 저장 완료")

                    except ValueError as e:
                        # 행렬 생성 불가능 오류(nnz_ratio가 너무 높을 때) 처리
                        print(f"조건 n={n}, W_ratio={W_ratio}, nnz={nnz}, diag={d} 에서 행렬 생성 불가: {e}")
                        # 이 경우 파일에 저장하지 않고 다음 조건으로 넘어갑니다.
                    except Exception as e:
                        print(f"기타 오류 발생: {e}")

print(f"\n Done Computing, Results are on '{file}' ")
