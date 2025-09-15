def plot_fig08():
  n       = 80        # number of states
  n_grid  = 100       # number of grid points for potential
  n_trial = 100       # number of trials
  n_each  = 100-1     # times steps for each trial
  dt      = 0.001     # delta t
  sigma = 0.8         # noise intensity
  c1 = 0.0005         # regularization parameter 1
  c2 = 0.0005         # regularization parameter 2
  dy = -1/n_each/dt
  epsilon = 1e-10
  np.random.seed(123)

  # potential function
  xx_arr = np.linspace(0, 1, n_grid)
  X, Y = np.meshgrid(xx_arr, xx_arr)
  yy_arr = xx_arr.reshape(-1,1)
  U = np.clip(2*yy_arr-1,0,None) * np.cos(2*np.pi*xx_arr) +\
    (1-np.abs(2*yy_arr-1)) * np.cos(2*np.pi*2*xx_arr) +\
    np.clip(1-2*yy_arr,0,None) * np.cos(2*np.pi*4*xx_arr)

  # time series
  v0 = np.array([0.5,1])
  sigma2  = sigma * dt**0.5
  def f(v):
    x, y = v
    dx = 2 * np.pi * (np.max([2*y-1,0]) * 1 * np.sin(2*np.pi*x) +
                      (1-np.abs(2*y-1)) * 2 * np.sin(2*np.pi*2*x) +
                      np.max([1-2*y,0]) * 4 * np.sin(2*np.pi*4*x))
    return np.array([dx, dy])
  v_list = []
  for _ in range(n_trial):
    v = v0.copy()
    out_list = [v]
    for _ in range(n_each):
      v = v + f(v) * dt + [sigma2 * np.random.randn(), 0]
      v = np.clip(v, 0, 1)
      out_list.append(v)
    v_list.append(np.array(out_list))
  v_arr = np.concatenate(v_list)

  # discretized time series
  model = KMeans(n_clusters=n, random_state=0, n_init='auto')
  model.fit(v_arr)
  v_df = pd.DataFrame(v_arr, columns=list('xy'))
  v_df['xd'] = model.predict(v_arr)
  v_df['trial'] = np.repeat(np.arange(n_trial), n_each+1)
  c_df = pd.DataFrame(model.cluster_centers_, columns=list('xy'))

  # voronoi partition
  v_dummy = np.array([[-1,-1],[-1,2],[2,-1],[2,2]])
  vor = Voronoi(np.vstack([c_df[['x','y']], v_dummy]))

  # transition matrix
  df = pd.DataFrame()
  df['src'] = v_df.xd.values[:-1]
  df['dst'] = v_df.xd.values[1:]
  df = df.value_counts().unstack().fillna(0).T
  df += epsilon
  df /= df.sum()
  A = df.values
  p_arr = calc_stationary(A)

  # no reset
  if False:
    df2 = pd.DataFrame()
    df2['src'] = v_df.groupby('trial').head(-1).xd.values
    df2['dst'] = v_df.groupby('trial').tail(-1).xd.values
    df2 = df2.value_counts().unstack().fillna(0).T
    df2 += epsilon
    df2 /= df2.sum()
    A2 = df2.values
    p2_arr = calc_stationary(A2)

  # control
  r_arr = -np.ones(n)
  idx1 = c_df[c_df.x < 1/4].sort_values('y').index[0]
  idx2 = c_df[(c_df.x > 1/4) & (c_df.x < 1/2)].sort_values('y').index[0]
  idx3 = c_df[(c_df.x > 1/2) & (c_df.x < 3/4)].sort_values('y').index[0]
  idx4 = c_df[(c_df.x > 3/4)].sort_values('y').index[0]
  r1_arr = r_arr.copy()
  r2_arr = r_arr.copy()
  r3_arr = r_arr.copy()
  r4_arr = r_arr.copy()
  r1_arr[idx1] = 1
  r2_arr[idx2] = 1
  r3_arr[idx3] = 1
  r4_arr[idx4] = 1
  dist_df = pd.DataFrame(distance.squareform(distance.pdist(c_df)))
  exclude_list = dist_df[dist_df>0.5].stack().index.to_list()
  h_arr = np.array([0.5, 0.8, 0.9])
  Ap1, F = calc_A_prime(A, r1_arr, h_arr, c1=c1, c2=c2,
                        exclude_list=exclude_list)
  Ap2, F = calc_A_prime(A, r2_arr, h_arr, c1=c1, c2=c2,
                        exclude_list=exclude_list)
  Ap3, F = calc_A_prime(A, r3_arr, h_arr, c1=c1, c2=c2,
                        exclude_list=exclude_list)
  Ap4, F = calc_A_prime(A, r4_arr, h_arr, c1=c1, c2=c2,
                        exclude_list=exclude_list)
  q1_arr = calc_stationary(A+Ap1)
  q2_arr = calc_stationary(A+Ap2)
  q3_arr = calc_stationary(A+Ap3)
  q4_arr = calc_stationary(A+Ap4)
  max_val = np.concatenate([q1_arr, q2_arr, q3_arr, q4_arr]).max()

  # plot
  fig, axes = plt.subplots(figsize=(12,12), ncols=3, nrows=3)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_label2 = dict(fontsize=16, ha='center', va='center', c='w',
                    fontweight='bold')

  def draw_voronoi(ax, vor, p_arr):
    ax.scatter(*vor.points[:n].T, s=5, c='k', zorder=10)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    for i in range(n):
      pos = vor.vertices[vor.regions[vor.point_region[i]]]
      ax.fill(*pos.T, alpha=p_arr[i]/max_val, color='C0')
      ax.fill(*pos.T, color='none', ec='k', lw=1, zorder=10)

  def draw_arrow(ax, Ap):
    for i in range(n):
      for j in range(n):
        if (i==j) or (Ap[i,j]==0):
          continue
        x1, y1 = c_df.loc[j]
        x2, y2 = c_df.loc[i]
        ax.arrow(x1, y1, x2-x1, y2-y1, color='C3', head_width=0.02,
                 length_includes_head=True)

  ax = axes[0]
  i_arr = np.arange(n_grid)+0.5
  sns.heatmap(U, ax=ax, cmap='viridis', vmin=-1, vmax=1)
  format_colorbar(ax)
  ax.invert_yaxis()
  ax.set_xticks([0.5, n_grid//2-0.5, n_grid-0.5],[0,0.5,1],
                rotation=0)
  ax.set_yticks([0.5, n_grid//2-0.5, n_grid-0.5],[0,0.5,1])
  for i in range(4):
    ax.text((2*i+1)/8, 1/16, str(i+1), transform=ax.transAxes, **kws_label2)
  ax.text(0.5, 15/16, '*', transform=ax.transAxes, **kws_label2)
  ax.set_title('potential function', **kws_title)
  ax.text(-0.18, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  for trial, sub_df in v_df.groupby('trial'):
    ax.plot(sub_df.x, sub_df.y, lw=1, c='C0')
  ax.set_xlim((0,1))
  ax.set_ylim((0,1) )   
  ax.set_title('trajectories', **kws_title)
  ax.text(-0.18, 1.02, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  draw_voronoi(ax, vor, p_arr)
  ax.set_title('K-means partition', **kws_title)
  ax.text(-0.18, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[3]
  draw_voronoi(ax, vor, q1_arr)
  draw_arrow(ax, Ap1)
  ax.set_title('control to target 1', **kws_title)
  ax.text(-0.18, 1.02, 'D', transform=ax.transAxes, **kws_label)

  ax = axes[4]
  draw_voronoi(ax, vor, q2_arr)
  draw_arrow(ax, Ap2)
  ax.set_title('control to target 2', **kws_title)
  ax.text(-0.18, 1.02, 'E', transform=ax.transAxes, **kws_label)

  ax = axes[5]
  draw_voronoi(ax, vor, q3_arr)
  draw_arrow(ax, Ap3)
  ax.set_title('control to target 3', **kws_title)
  ax.text(-0.18, 1.02, 'F', transform=ax.transAxes, **kws_label)

  ax = axes[6]
  draw_voronoi(ax, vor, q4_arr)
  draw_arrow(ax, Ap4)
  ax.set_title('control to target 4', **kws_title)
  ax.text(-0.18, 1.02, 'G', transform=ax.transAxes, **kws_label)

  axes[7].axis('off')
  axes[8].axis('off')

  for ax in axes:
    ax.tick_params(length=5, width=1)
    format_spine(ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return vor

if __name__ == '__main__':
  hoge = plot_fig08()
