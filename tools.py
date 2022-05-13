import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm

def integrate(A, X0, dt, N):
    """
        Computes the discretization of system (dot)X = A X over intervals of length dt.
        Returns X0, X1, ... X(n-1)
    """
    e = expm(A*dt)
    res = np.zeros((N, 2), dtype=complex)
    res[0, :] = X0
    for i in range(1, N):
        res[i, :] = e @ res[i - 1, :]
    return res

class Evolutions:
    """
        Handles computation and animation simultaneously
    """
    def __init__(self, A, dt, N, slicing=0):
        self.prev_ani = None
        self.A = np.zeros(A.shape, dtype=complex)
        self.A[:] = A
        self.dt = dt
        self.N = N
        self.slicing = slicing
        
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.fig.suptitle(r'Time frame $[0, '+ f'{self.N * self.dt:.2f}' + r']$')

        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        
        # Get eigenvalues and eigenvectors
        eigvals, eigvec = np.linalg.eig(self.A)
        self.ax.quiver([0, 0], [0, 0], eigvec[0, :], eigvec[1, :], linewidth=1.)
        eig0 = eigvec[:, 0]
        eig1 = eigvec[:, 1]
        for v in [eig0, eig1]:
            n_v = v / np.linalg.norm(v)
            self.ax.plot([-2*n_v[0], 2*n_v[0]], [-2*n_v[1], 2*n_v[1]], linestyle='-.', color='gray', linewidth=1.)

        print("Eigenvalues : ")
        print(list(eigvals))
    
    def handle_event(self, event):
        self.new(np.array((event.xdata, event.ydata)))
    
    def new(self, X0):
        
        pos = integrate(self.A, X0, self.dt, self.N)
        line, = self.ax.plot([], [])
        def animate(i):
            last = 0
            if self.slicing != 0:
                if i>self.slicing:
                    last = i-self.slicing
            line.set_data(pos[last:i,0], pos[last:i,1])#, color=[0.99*j for j in range(i)].reverse())
            return line,

        ani = animation.FuncAnimation(self.fig, animate, frames=self.N, blit=True, interval=5, repeat=False)
        self.prev_ani = ani
    
    def play(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.handle_event)

