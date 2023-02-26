import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import sys

plt.rcParams['font.size'] = 30
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.usetex'] = True


class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)




theta = np.linspace(0, np.pi, 300)
phi = np.linspace(0, 2*np.pi, 300)
theta, phi = np.meshgrid(theta, phi)
theta, phi = theta.ravel(), phi.ravel()
mesh_x, mesh_y = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)
triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
x, y, z = np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)
x, y, z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

beta = float(sys.argv[1])
gamma = lambda b: 1/(np.sqrt(1-b**2))
# Defining a custom color scalar field
vals = (1/(1 - beta*np.cos(theta))**3) * (1 - (np.sin(theta)*np.cos(phi))**2 / (gamma(beta)*(1-beta*np.cos(theta)))**2) 
colors = np.mean(vals[triangles], axis=1)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('viridis')
triang = mtri.Triangulation(x, y, triangles)
collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
collec.set_array(colors)
ax.set_aspect("equal")

ax.arrow3D(0,0,0,
           0,0,1.5,
           mutation_scale=20,
           arrowstyle="-|>", fc='k')
ax.annotate3D(r'$\hat{z}$', (0, 0, 1.5), xytext=(3, 3), textcoords='offset points')
ax.arrow3D(0,0,0,
           1.5,0,0,
           mutation_scale=20,
           arrowstyle="-|>", fc='k')
ax.annotate3D(r'$\hat{x}$', (1.7, 0, 0), xytext=(3, 3), textcoords='offset points')
ax.arrow3D(0,0,0,
           0,1.5,0,
           mutation_scale=20,
           arrowstyle="-|>", fc='k')
ax.annotate3D(r'$\hat{y}$', (0, 1.5, 0), xytext=(3, 3), textcoords='offset points')
ax.arrow3D(0,0,0,
           0,0,0.7,
           mutation_scale=20,
           arrowstyle="-|>", fc='r')
ax.annotate3D(r'$\hat{\beta}$', (0, 0, 0.7), xytext=(3, 3), textcoords='offset points')
ax.arrow3D(0,0,0,
           0.7,0,0,
           mutation_scale=20,
           arrowstyle="-|>", fc='r')
ax.annotate3D(r'$\hat{\dot{\beta}}$', (0.7, 0, 0.1), xytext=(3, 3), textcoords='offset points')


#Draw circular patch 
zeta = np.linspace(0, 2*np.pi, 100)
r = 1.5 
x = r*np.cos(zeta) + r
y = r*np.zeros(100)
z = r*np.sin(zeta)
ax.plot(x, y, z, lw=3)

ax.set(title=fr'$\beta={beta}$')
ax.view_init(elev=35, azim=50)
plt.show()

