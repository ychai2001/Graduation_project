from make_sys import *

ns = [10000, 20000, 30000, 40000]
Ws = [100, 200, 400, 800]
nnzs = [0.005, 0.01, 0.05, 0.1]
diags = [1.5, 2, 4, 8]

file = 'test.csv'

with open(file, 'w') as f:
    # 1. 파일 헤더 (Header) 쓰기
    # n, W, nnz, d, t_cpu, t_gpu 순서로 저장
    header = "n, W, nnz_ratio, diag, t_cpu, t_gpu\n"
    f.write(header)

    for n in ns:
        for W in Ws:
            for nnz in nnzs:
                for d in diags:
                    try:
                        a = make_A(n, W, nnz, diag=d)
                        t_cpu, t_gpu = cal_time(a)

                        data_row = f"{n},{W},{nnz},{d},{t_cpu},{t_gpu}\n"
                        f.write(data_row)
                        print("저장 완료")

                    except ValueError as e:
                        # 행렬 생성 불가능 오류(nnz_ratio가 너무 높을 때) 처리
                        print(f"❌ 조건 {n}, W={W}, nnz={nnz}, diag={d} 에서 행렬 생성 불가: {e}")
                        # 이 경우 파일에 저장하지 않고 다음 조건으로 넘어갑니다.
                    except Exception as e:
                        print(f"🚨 기타 오류 발생: {e}")

