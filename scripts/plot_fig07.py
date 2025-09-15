def plot_fig07():
  n_list  = [20,40,60,80,100]  # number of states
  n_step  = 10**4     # time steps
  d       = 1.5       # data range
  dt      = 0.05      # delta t
  sigma   = 1.0       # noise intensity
  c1 = 0.003          # regularization parameter 1
  c2 = 0.003          # regularization parameter 2
  epsilon = 1e-10
  np.random.seed(12345)

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

  # loop
  out_list = []
  t_list = []
  for n in n_list:
    # time
    t0 = time.time()

    ## K-means
    model = KMeans(n_clusters=n, random_state=0, n_init='auto')
    model.fit(v_arr)
    v_df = pd.DataFrame(v_arr, columns=list('xy'))
    v_df['xd'] = model.predict(v_arr)
    c_df = pd.DataFrame(model.cluster_centers_, columns=list('xy'))

    # voronoi partition
    v_dummy = 2 * d * np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
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

    # control
    r_arr = (c_df.x<0).astype(int) + (c_df.y<0).astype(int) - 1
    h_arr = np.array([0.5, 0.8, 0.9])
    Ap, F = calc_A_prime(A, r_arr, h_arr, c1=c1, c2=c2)
    q_arr = calc_stationary(A+Ap)

    # save
    out_list.append(dict(n=n, vor=vor, q_arr=q_arr, Ap=Ap, c_df=c_df))
    t_list.append(time.time() - t0)

  # plot
  fig, axes = plt.subplots(figsize=(12,8), ncols=3, nrows=2)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  label_list = list('ABCDE')

  def draw_voronoi(ax, vor, p_arr, max_val):
    ax.scatter(*vor.points[:n].T, s=5, c='k', zorder=10)
    ax.set_xlim((-d,d))
    ax.set_ylim((-d,d))
    for i in range(n):
      pos = vor.vertices[vor.regions[vor.point_region[i]]]
      ax.fill(*pos.T, alpha=p_arr[i]/max_val, color='C0')
      ax.fill(*pos.T, color='none', ec='k', lw=1, zorder=10)

  def draw_arrow(ax, Ap, c_df):
    for i in range(n):
      for j in range(n):
        if (i==j) or (Ap[i,j]==0):
          continue
        x1, y1 = c_df.loc[j]
        x2, y2 = c_df.loc[i]
        ax.arrow(x1, y1, x2-x1, y2-y1, color='C3', head_width=0.08,
                 length_includes_head=True)

  for ax, res_dict, label in zip(axes, out_list, label_list):
    n = res_dict['n']
    vor = res_dict['vor']
    Ap = res_dict['Ap']
    c_df = res_dict['c_df']

    draw_voronoi(ax, vor, q_arr, q_arr.max())
    draw_arrow(ax, Ap, c_df)
    ax.set_title(f'K={n}', **kws_title)
    ax.text(-0.18, 1.02, label, transform=ax.transAxes, **kws_label)

    ax.tick_params(length=5, width=1)
    format_spine(ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')

  ax = axes[-1]
  ax.plot(t_list, 'o-')
  ax.set_xticks(np.arange(len(n_list)), n_list)
  ax.set_xlabel('$K$')
  ax.set_ylabel('time (s)')
  ax.set_yscale('log')
  ax.set_title(f'computation time', **kws_title)
  ax.text(-0.18, 1.02, 'F', transform=ax.transAxes, **kws_label)
  ax.tick_params(length=5, width=1)
  format_spine(ax)

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return t_list

if __name__ == '__main__':
  hoge = plot_fig07()
