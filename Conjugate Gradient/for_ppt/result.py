import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt

result = 'result.csv'
h_input = ['n', 'W_ratio', 'nnz_ratio', 'diag']
h_output = 'su_gpu'
headers = h_input + [h_output]

df = pd.read_csv(result, usecols=headers, skipinitialspace=True)

x = df[h_input].values
y = df[h_output].values

f = LinearNDInterpolator(x, y)

ns = np.linspace(4000, 20000, 11)
nnzs = np.linspace(0.005, 0.1, 101)
ws = np.linspace(0.05, 0.5, 101)
#diag = np.linspace(0.5, 2, 11)
NNZ, W = np.meshgrid(nnzs, ws)

diag_fixed = 1

rows, cols = W.shape
for n in ns:
    Z_predict = np.zeros(W.shape)

    for i in range(rows):
        for j in range(cols):
            # 현재 w와 nnz 값
            current_w = W[i, j]
            current_nnz = NNZ[i, j]
            
            # 4차원 입력 배열 (LinearNDInterpolator는 2차원 배열을 기대합니다: (1, 4))
            x_new = np.array([[n, current_w, current_nnz, diag_fixed]])
            
            # 보간 값 계산 및 배열에 저장
            predicted_value = f(x_new)
            
            # nan 값이 나올 경우, 그래프를 망치지 않기 위해 0 또는 다른 상수로 대체 가능
            # (LinearNDInterpolator가 데이터 범위를 벗어난 곳에서 nan을 반환하므로)
            if np.isnan(predicted_value[0]):
                Z_predict[i, j] = np.nan # 외삽 영역은 0으로 처리 (혹은 다른 적절한 값)
            else:
                Z_predict[i, j] = predicted_value[0] - 1   #1이 가속 기준임
    
    plt.figure(figsize=(10, 8))

    contour_filled = plt.contourf(
        W, 
        NNZ, 
        Z_predict, 
        levels=15,          # 등고선의 단계를 15개로 설정
        cmap='coolwarm'     # 양수/음수를 잘 표현하는 색상 맵 사용
    )

    plt.contour(
        W, 
        NNZ, 
        Z_predict, 
        levels=[1],         # Z_predict=1 인 지점만 선으로 표시
        colors='red',       # 다른 색상으로 명확하게 구분
        linewidths=2.5,     # 두께를 더 두껍게
        linestyles='-'
    )

    # 2. (선택 사항) 등고선(Contour)을 위에 겹쳐 그리기
    # 등고면 위에 선을 추가하여 경계를 더 명확하게 보여줄 수 있습니다.
    # colors='k' : 선 색상을 검정(black)으로 지정합니다.
    contour_lines = plt.contour(
        W, 
        NNZ, 
        Z_predict, 
        levels=contour_filled.levels, # contourf와 동일한 레벨 사용
        colors='black',
        linewidths=0.5
    )

    # 3. 등고선에 라벨 값 추가 (선택 사항)
    # fmt='%1.1f' : 라벨 값의 소수점 첫째 자리까지 표시
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
    plt.xlabel('W_ratio (w)')
    plt.ylabel('nnz_ratio (nnz)')
    plt.title(f'Contour Plot of Predicted su_gpu (n={int(n)}, diag={diag_fixed})')
    cbar = plt.colorbar(contour_filled)
    cbar.set_label('Predicted su_gpu Value')


    plt.savefig(f'speedups/su_cupy_n{int(n)}_diag{diag_fixed}', dpi=300)
