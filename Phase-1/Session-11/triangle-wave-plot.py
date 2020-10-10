import numpy as np
import matplotlib.pyplot as plt

num_cycles = 6
step_size = 20
lr_min = 0.001
lr_max = 0.010
total_iters = step_size * 2 * num_cycles


def get_lr(step):
    cycle = np.floor(1 + step / (2 * step_size))
    x = np.abs(step / step_size - 2 * cycle + 1)
    return lr_min + (lr_max - lr_min) * (1 - x)


x = np.linspace(0, total_iters, 1000)
y = np.array([get_lr(iter) for iter in x])


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y)
ax.set_xlabel("step")
ax.set_ylabel("lr")
ax.grid()
fig.savefig("./images/triangle-wave.png", bbox_inches="tight", pad_inches=0.25)