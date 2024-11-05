import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = [1, 2, 3, 4, 5]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

def animate(frame):
    ax.clear()
    ax.plot(data[:frame+1])

anim = FuncAnimation(fig, animate, frames=len(data), interval=1000)

plt.show()

