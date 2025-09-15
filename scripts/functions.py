import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ot
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

sns.set_context('talk', font_scale=0.8)

def z_score(df):
  return (df - df.mean())/df.std()

def calc_A_theory_1d(n, d=1.8, dt=0.02, sigma=1.2):
  sigma2  = sigma * dt**0.5
  x2_arr  = np.linspace(-d, d, n+1)
  mid_arr = (x2_arr[:-1] + x2_arr[1:]) / 2
  fx_arr  = -4 * mid_arr**3 + 4 * mid_arr
  models  = stats.norm(loc=mid_arr + dt*fx_arr, scale=sigma2)
  A = models.cdf(x2_arr[1:].reshape(-1,1)) - \
      models.cdf(x2_arr[:-1].reshape(-1,1))
  A += 1e-10
  A /= A.sum(axis=0)
  return A

def calc_stationary(A):
  w_arr, V = np.linalg.eig(A)
  p_arr = V[:,np.argmax(w_arr)].real
  p_arr /= p_arr.sum()
  return p_arr

def calc_horizon(A, p0_arr, horizon):
  return np.linalg.matrix_power(A, horizon).dot(p0_arr)

def calc_objective(A, Ap, r_arr, c1, c2, p0_arr, horizon):
  if horizon is None:
    A_mask =np.ma.masked_where(A==0, A)
    return (r_arr * calc_stationary(A+Ap)).sum() -c1 * (Ap != 0).sum() \
        -c2 * np.abs(np.log2((A+Ap)/A_mask)).sum()
  else:
    return (r_arr * calc_horizon(A+Ap, p0_arr, horizon)).sum() \
        -c1 * (Ap != 0).sum() -c2 * np.abs(np.log2((A+Ap)/A)).sum()

def calc_A_prime(A, r_arr, h_arr, c1=0.01, c2=0.01, perc=80, verbose=True,
                 exclude_list=[], p0_arr=None, horizon=None):
  n = len(A)
  Ap_gbest = np.zeros_like(A)
  F_best = calc_objective(A, Ap_gbest, r_arr, c1, c2, p0_arr, horizon)

  A_tmp = A.copy()
  np.fill_diagonal(A_tmp, 0)
  th = np.percentile(A_tmp, perc)
  out_list = []
  for i in range(n):
    for j in range(n):
      if i==j:
        continue
      if (i,j) in exclude_list:
        continue
      if A[i,j] < th:
        continue
      Ap = np.zeros_like(A)
      Ap[i,j] = -0.5 * A[i,j]
      Ap[j,j] =  0.5 * A[i,j]
      F = calc_objective(A, Ap, r_arr, 0, c2, p0_arr, horizon)
      if F > F_best:
        out_list.append(dict(i=i,j=j,F=F))
  df = pd.DataFrame(out_list).sort_values('F', ascending=False)
  # df = df[df.F > df.F.quantile(0.5)]  # not much effective
  ij_arr = df[['i','j']].values
  if verbose:
    print(f'target elements: {len(df)} ({100*len(df)/n**2:.1f} %)')

  while True:
    # single element
    i_best = None
    j_best = None
    h_best = None
    Ap_best = None
    flag = False
    for i,j in ij_arr:
      for h in h_arr:
        Ap = Ap_gbest.copy()  # move this outside h-loop changes result (why?)
        Ap[i,j] = -h * A[i,j]
        Ap[j,j] -= Ap[:,j].sum()
        F = calc_objective(A, Ap, r_arr, c1, c2, p0_arr, horizon)
        if F > F_best:
          flag = True
          i_best, j_best, h_best, F_best, Ap_best = i, j, h, F, Ap
    if flag == False:
      break
    if verbose:
      print(f'i={i_best:>2d}, j={j_best:>2d}, h={h_best:.2f}, '\
            f'F={F_best:.5f}')
    Ap_gbest = Ap_best
  return Ap_gbest, F_best

def calc_smoothing(A, D, gamma):
  n = len(A)
  K = np.exp(-gamma * D**2)
  out_list = []
  for i in range(n):
    out_list.append((K[i]*A).sum(axis=1) / K[i].sum())
  A_smooth = np.array(out_list).T
  return A_smooth

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

def format_colorbar(ax):
  cb = ax.collections[0].colorbar
  cb.ax.tick_params(length=3, width=1)
  cb.outline.set_linewidth(1)

def format_spine(ax):
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
