def sub_plot_fig08(f, v0, n):
  n_step  = 2*10**4   # times steps
  n_cut   = 1000      # times steps to be cut
  dt      = 0.01      # delta t
  epsilon = 1e-10

  # simulation
  v = v0
  out_list = [v]
  for _ in range(n_step):
    v1 = f(v)
    v2 = f(v+0.5*dt*v1)
    v3 = f(v+0.5*dt*v2)
    v4 = f(v+dt*v3)
    v = v + dt*(v1+2*v2+2*v3+v4)/6
    out_list.append(v)
  v_df = pd.DataFrame(out_list[n_cut:], columns=list('xyz'))

  # discretized time series
  model = KMeans(n_clusters=n, random_state=0, n_init='auto')
  model.fit(v_df)
  v_df['xd'] = model.predict(v_df)
  c_df = pd.DataFrame(model.cluster_centers_, columns=list('xyz'))

  # sort index
  Z = linkage(c_df, metric='euclidean', method='ward')
  sr = pd.Series(leaves_list(Z).argsort())
  v_df['xd'] = v_df.xd.replace(sr)
  c_df = c_df.iloc[leaves_list(Z)].reset_index(drop=True)
  
  # transition matrix
  df = pd.DataFrame()
  df['src'] = v_df.xd.values[:-1]
  df['dst'] = v_df.xd.values[1:]
  df = df.value_counts().unstack().fillna(0).T
  df += epsilon
  df /= df.sum()
  A = df.values
  p_arr = calc_stationary(A)

  # return
  return v_df, c_df, A, p_arr

def plot_fig08():
  n       = 50        # number of states
  c1_lo   = 0.01      # regularization parameter 1, Lorenz
  c2_lo   = 0.01      # regularization parameter 2, Lorenz
  c1_rs   = 0.002     # regularization parameter 1, Rossler
  c2_rs   = 0.002     # regularization parameter 2, Rossler
  n_color = 10        # number of colors

  # Lorentz attractor
  rho = 28
  sigma = 10
  beta = 8/3
  v0_lo = np.array([0,1,10])
  def f_lo(v):
    x, y, z = v
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

  # Rossler attractor
  a = 0.1
  b = 0.1
  c = 14
  v0_rs = np.array([10,10,0])
  def f_rs(v):
    x, y, z = v
    return np.array([-y-z, x+a*y, b+z*(x-c)])

  # sub func
  v1_df, c1_df, A1, p1_arr = sub_plot_fig08(f_lo, v0_lo, n)
  v2_df, c2_df, A2, p2_arr = sub_plot_fig08(f_rs, v0_rs, n)

  # control
  r1_arr = 2*(c1_df.x>0).astype(int) - 1
  r2_arr = 2*(c2_df.z<1).astype(int) - 1
  h_arr = np.array([0.5])
  Ap1, F = calc_A_prime(A1, r1_arr, h_arr, c1=c1_lo, c2=c2_lo)
  Ap2, F = calc_A_prime(A2, r2_arr, h_arr, c1=c1_rs, c2=c2_rs)
  q1_arr = calc_stationary(A1+Ap1)
  q2_arr = calc_stationary(A2+Ap2)

  # plot
  fig = plt.figure(figsize=(10,8))
  kws_ax = dict(projection='3d', computed_zorder=False)
  ax1 = fig.add_subplot(221, **kws_ax)
  ax2 = fig.add_subplot(222, **kws_ax)
  ax3 = fig.add_subplot(223, **kws_ax)
  ax4 = fig.add_subplot(224, **kws_ax)
  axes = np.array([ax1, ax2, ax3, ax4])
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_scat = dict(s=30, cmap='Blues', ec='k', lw=1)
  kws_tick = dict(ha='center', va='center')
  c_arr = sns.color_palette('tab10', n_color)

  def draw_arrow(ax, Ap, c_df):
    for i in range(n):
      for j in range(n):
        if (i==j) or (Ap[i,j]==0):
          continue
        x1, y1, z1 = c_df.loc[j]
        x2, y2, z2 = c_df.loc[i]
        ax.add_artist(Arrow3D(x1,y1,z1,x2-x1,y2-y1,z2-z1, arrowstyle='-|>',
                              mutation_scale=10, color='C3'))

  ax = axes[0]
  ax.plot(v1_df.x, v1_df.y, v1_df.z, lw=1, color='0.8')
  for i in range(n):
    sub_df = v1_df[v1_df.xd==i]
    ax.scatter(sub_df.x, sub_df.y, sub_df.z, s=1, c=[c_arr[i%n_color]],
               zorder=10)
  ax.scatter(c1_df.x, c1_df.y, c1_df.z, s=10, c='k', zorder=20)
  ax.set_title('Lorenz attractor (K-means partition)')
  ax.text(-45, 0, 52, 'A', **kws_label)

  ax = axes[1]
  ax.scatter(c1_df.x, c1_df.y, c1_df.z, c=q1_arr, **kws_scat)
  draw_arrow(ax, Ap1, c1_df)
  ax.set_title('intervention to Lorenz attractor')
  ax.text(-45, 0, 52, 'B', **kws_label)

  ax = axes[2]
  ax.plot(v2_df.x, v2_df.y, v2_df.z, lw=1, color='0.8')
  for i in range(n):
    sub_df = v2_df[v2_df.xd==i]
    ax.scatter(sub_df.x, sub_df.y, sub_df.z, s=1, c=[c_arr[i%n_color]],
               zorder=10)
  ax.scatter(c2_df.x, c2_df.y, c2_df.z, s=10, c='k', zorder=20)
  ax.set_title('R\u00f6ssler attractor (K-means partition)')
  ax.text(-37, -20, 46, 'C', **kws_label)

  ax = axes[3]
  ax.scatter(c2_df.x, c2_df.y, c2_df.z, c=q2_arr, **kws_scat)
  draw_arrow(ax, Ap2, c2_df)
  ax.set_title('intervention to R\u00f6ssler attractor')
  ax.text(-37, -20, 46, 'D', **kws_label)

  for ax in axes:
    ax.tick_params(pad=-0.5)
    ax.set_xlabel('$x$', labelpad=-5)
    ax.set_ylabel('$y$', labelpad=-5)
    ax.set_zlabel('$z$', labelpad=-5)
    ax.grid(False)
    ax.set_box_aspect((1,1,1), zoom=1.2)
    ax.view_init(elev=15, azim=-60)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
      #axis.line.set_linewidth(1)
      pass

  for ax in axes[:2]:
    ax.set_xticks([-15,0,15], ['$-15$',0,15], **kws_tick)
    ax.set_yticks([-20,0,20], ['$-20$',0,20], **kws_tick)
    ax.set_zticks([10,20,30,40], [10,20,30,40], **kws_tick)
    ax.set_xlim((-20,20))
    ax.set_ylim((-26,26))
    ax.set_zlim((2,47))
    
  for ax in axes[2:]:
    ax.set_xticks([-20,0,20], ['$-20$',0,20], **kws_tick)
    ax.set_yticks([-20,0,20], ['$-20$',0,20], **kws_tick)
    ax.set_zticks([0,10,20,30], [0,10,20,30], **kws_tick)
    ax.set_xlim((-23,23))
    ax.set_ylim((-23,23))
    ax.set_zlim((-2,38))
    
  fig.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95,
                      hspace=0.25)
  fig.show()
  fig.savefig('tmp.png')
  return v1_df

if __name__ == '__main__':
  hoge = plot_fig08()
