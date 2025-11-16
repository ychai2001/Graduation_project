from make_sys import *

# In constant size(20000) & diag=1.2
ns = [10000, 20000, 30000, 40000, 50000]

# In constant size(20000) & diag=1.2
ws = [100, 500, 1000, 5000, 10000]
W = 100

ts_cpu=[]
ts_gpu=[]

for n in ns:
    a = make_A(n, W)
    t_cpu, t_gpu = cal_time(a)
    ts_cpu.append(t_cpu)
    ts_gpu.append(t_gpu)

plt.plot(ns, ts_cpu, label='cpu')
plt.plot(ns, ts_gpu, label='gpu')
plt.title('Computing time vs Grid size n')
plt.legend()
plt.show()