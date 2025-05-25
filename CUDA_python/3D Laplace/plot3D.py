import numpy as np
from mayavi import mlab
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import pyplot as plt
import time

ti_serial = np.load("ti_serial.npy")
time_serial = np.load("time_serial.npy")

# # 3D 등치면 시각화
# mlab.contour3d(ti_serial, contours=10, opacity=0.5)
# mlab.show()

src = mlab.pipeline.scalar_field(ti_serial)
contour = mlab.pipeline.contour_surface(src, contours=[0.0], opacity=1)
mlab.contour3d(ti_serial, contours=11, opacity=0.1)

# 등치면 값 변화 애니메이션
for val in np.linspace(100, 300, 21):
    contour.contour.contours = [val]
    mlab.title(f"Isosurface value: {val:.2f}", size=0.5)
    mlab.process_ui_events()
    time.sleep(0.15)
mlab.show()

# fig, ax = plt.subplots()
# cax = ax.imshow(ti_serial[:,0,:], cmap='viridis', origin='lower', vmin= 100, vmax= 300)
# fig.colorbar(cax)
# title = ax.set_title("y = 0")

# def update(frame):
#     cax.set_array(ti_serial[:,frame,:])
#     title.set_text(f"y = {frame}")
#     return cax, title

# ani = FuncAnimation(fig, update, frames=range(64), interval=200, blit=False)
# HTML(ani.to_jshtml())