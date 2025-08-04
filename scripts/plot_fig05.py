def plot_fig05():
  n_grid  = 10        # number of grid points per axis
  n_grid2 = 100       # number of grid points per axis, for potential
  n       = n_grid**2 # number of states
  n_step  = 10**4     # time steps
  n_show  = 2000      # time steps to show for trajectory
  d       = 1.5       # data range
  dt      = 0.05      # delta t
  sigma   = 1.0       # noise intensity
  c1 = 0.005          # regularization parameter 1
  c2 = 0.005          # regularization parameter 2
  epsilon = 1e-10
  np.random.seed(123)

  # grid
  xx_arr = np.linspace(-d, d, n_grid)  
  X, Y = np.meshgrid(xx_arr, xx_arr)

  # potential function
  # NOTE: U = V**4 - 2*V**2 +4*W**2, V=(X+Y)/sqrt(2), W=(X-Y)/sqrt(2)
  xx2_arr = np.linspace(-d, d, n_grid2)
  X2, Y2 = np.meshgrid(xx2_arr, xx2_arr)
  U = 0.25*(X2+Y2)**4 -(X2+Y2)**2 + 2*(X2-Y2)**2

  # time series
  v = np.zeros(2)
  sigma2  = sigma * dt**0.5
  def f(v):
    x, y = v
    return np.array([-(x+y)**3 -2*x +6*y, -(x+y)**3 -2*y +6*x])
  out_list = []
  for _ in range(n_step):
    v = v + f(v) * dt + sigma2 * np.random.randn(2)
    out_list.append(v)
  v_arr = np.array(out_list)

  # discretized time series
  vd_arr = (n_grid * (v_arr + d) / (2*d)).astype(int)
  vd_arr = np.clip(vd_arr, 0, n_grid-1)
  xd_arr = vd_arr[:,0] + vd_arr[:,1] * n_grid

  # empirical distribution
  p_sr = pd.Series(xd_arr).value_counts().sort_index()
  p_sr /= p_sr.sum()
  p_sr = p_sr.reindex(np.arange(n)).fillna(0)
  P = p_sr.values.reshape(n_grid, n_grid)

  # transition matrix
  df = pd.DataFrame()
  df['src'] = xd_arr[:-1]
  df['dst'] = xd_arr[1:]
  df = df.value_counts().unstack().fillna(0).T
  df = df.reindex(index=np.arange(n), columns=np.arange(n)).fillna(0)
  df += epsilon
  df /= df.sum()
  A = df.values
  P2 = calc_stationary(A).reshape(n_grid, n_grid)

  # control
  R = np.ones_like(X) - ((X>0).astype(int) + (Y>0).astype(int))
  r_arr = R.flatten()  
  h_arr = np.array([0.5, 0.8, 0.9])
  Ap, F = calc_A_prime(A, r_arr, h_arr, c1=c1, c2=c2)
  P3 = calc_stationary(A+Ap).reshape(n_grid, n_grid)
  print(P3.max())

  # plot
  fig, axes = plt.subplots(figsize=(12,8), ncols=3, nrows=2)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_hmap  = dict(cmap='Blues', xticklabels=1, yticklabels=1, vmin=0,
                   vmax=0.12)
  kws_cont = dict(levels=10, colors='w', linewidths=1)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  i_arr = np.arange(n_grid)+1
  i2_arr = np.arange(n_grid2)+0.5

  ax = axes[0]
  sns.heatmap(U, ax=ax, cmap='viridis')
  ax.contour(i2_arr, i2_arr, U, **kws_cont)
  format_colorbar(ax)
  ax.invert_yaxis()
  ax.set_xticks([0.5, n_grid2//2-0.5, n_grid2-0.5],[f'${-d}$',0,d],
                rotation=0)
  ax.set_yticks([0.5, n_grid2//2-0.5, n_grid2-0.5],[f'${-d}$',0,d])
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('potential function', **kws_title)
  ax.text(-0.15, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  ax.plot(*v_arr[:n_show].T, lw=1)
  ax.set_xlim((-d,d))
  ax.set_ylim((-d,d))
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('trajectory', **kws_title)
  ax.text(-0.15, 1.02, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  ax.plot(v_arr[:,0], lw=1, label='x')
  ax.plot(v_arr[:,1], lw=1, label='y')
  g = ax.legend(frameon=False, loc='upper center', ncols=2, markerscale=5)
  for h in g.legend_handles:
    h.set_linewidth(2)
  ax.set_ylim((-d,1.5*d))
  ax.set_xlabel('time step')
  ax.set_ylabel('$x,\ y$')
  ax.set_title('time series', **kws_title)
  ax.text(-0.15, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[3]
  sns.heatmap(P, ax=ax, **kws_hmap)
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('empirical distribution\n', **kws_title)
  ax.text(-0.15, 1.1, 'D', transform=ax.transAxes, **kws_label)

  ax = axes[4]
  sns.heatmap(P2, ax=ax, **kws_hmap)
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('predicted distribution\nwithout control', **kws_title)
  ax.text(-0.15, 1.1, 'E', transform=ax.transAxes, **kws_label)

  ax = axes[5]
  sns.heatmap(P3, ax=ax, **kws_hmap)
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('predicted distribution\nwith control', **kws_title)
  ax.text(-0.15, 1.1, 'F', transform=ax.transAxes, **kws_label)

  for ax in axes[3:]:
    ax.tick_params(length=0)
    format_colorbar(ax)
    format_spine(ax)
    ax.set_xticklabels(i_arr)
    ax.set_yticklabels(i_arr, rotation=0)
    ax.invert_yaxis()

  for ax in axes[:3]:
    ax.tick_params(length=5, width=1)
    format_spine(ax)

  for ax in axes[[5]]:
    for i in range(n):
      for j in range(n):
        if (i==j) or (Ap[i,j]==0):
          continue
        x1, y1 = j%n_grid, j//n_grid
        x2, y2 = i%n_grid, i//n_grid
        rx, ry = 0.3*(2 * np.random.rand(2)-1)
        x1, x2, y1, y2 = np.array([x1+rx,x2+rx,y1+ry,y2+ry])
        ax.arrow(x1+0.5, y1+0.5, x2-x1, y2-y1, color='C3', head_width=0.2,
                 length_includes_head=True, zorder=10)

  fig.tight_layout(h_pad=2)
  fig.show()
  fig.savefig('tmp.png')
  return P

if __name__ == '__main__':
  hoge = plot_fig05()
