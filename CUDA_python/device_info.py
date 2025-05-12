from numba import cuda
print(cuda.detect())

# 현재 사용 가능한 GPU 정보 가져오기
device = cuda.get_current_device()


print(f"GPU Name: {device.name}")
print(f"총 멀티프로세서 수 (Multiprocessors): {device.MULTIPROCESSOR_COUNT}")
print(f"각 블록당 최대 스레드 수: {device.MAX_THREADS_PER_BLOCK}")
print(f"각 블록의 최대 차원 (x, y, z): {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}")
print(f"각 그리드의 최대 차원 (x, y, z): {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}")
# MAX_THREADS_PER_MULTIPROCESSOR 속성이 존재하는지 확인 후 출력
if hasattr(device, "MAX_THREADS_PER_MULTIPROCESSOR"):
    print(f"한 멀티프로세서당 최대 스레드 수: {device.MAX_THREADS_PER_MULTIPROCESSOR}")
else:
    print("MAX_THREADS_PER_MULTIPROCESSOR 속성이 존재하지 않습니다.")

