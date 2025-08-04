def plot_fig02():
  n      = 20        # number of states
  n_step = 10**4     # number of time steps
  d      = 1.7       # data range
  dt     = 0.01      # delta t
  sigma  = 1.0       # noise intensity
  epsilon = 1e-10
  np.random.seed(123)

  # A. potential function
  xx_arr = np.linspace(-d, d, n)
  U_arr = xx_arr**4 - 2*xx_arr**2

  # B. time series
  x = 0
  sigma2  = sigma * dt**0.5
  def f(x):
    return -4 * x**3 +4 * x
  out_list = []
  for _ in range(n_step):
    x = x + f(x) * dt + sigma2 * np.random.randn()
    out_list.append(x)
  x_arr = np.array(out_list)

  # C. discretized time series
  xd_arr = (n * (x_arr + d) / (2*d)).astype(int)
  xd_arr = np.clip(xd_arr, 0, n-1) + 1

  # D. empirical distribution
  p_sr = pd.Series(xd_arr).value_counts().sort_index()
  p_arr = (p_sr / p_sr.sum()).values

  # E. transition matrix
  df = pd.DataFrame()
  df['src'] = xd_arr[:-1]
  df['dst'] = xd_arr[1:]
  df = df.value_counts().unstack().fillna(0).T
  df /= df.sum()
  A = df.values
  A += epsilon
  A /= A.sum(axis=0)

  # F. predicted distribution
  p2_arr = calc_stationary(A)

  # plot
  fig, axes = plt.subplots(figsize=(12,8), ncols=3, nrows=2)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_hmap  = dict(cmap='Blues', xticklabels=1, yticklabels=1, vmin=0)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  i_arr = np.arange(n)+1

  ax = axes[0]
  ax.plot(xx_arr, U_arr, 'o-')
  ax.set_xlabel('$x$')
  ax.set_ylabel('$U(x)$')
  ax.set_title('potential function', **kws_title)
  ax.text(-0.15, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  ax.plot(x_arr)
  ax.set_xlabel('time step')
  ax.set_ylabel('$x$')
  ax.set_title('time series', **kws_title)
  ax.text(-0.15, 1.02, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  ax.plot(xd_arr)
  ax.set_xlabel('time step')
  ax.set_ylabel('state number')
  ax.set_title('discretized time series', **kws_title)
  ax.text(-0.15, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[3]
  ax.bar(i_arr, p_arr)
  ax.set_xlabel('state number')
  ax.set_ylabel('probability')
  ax.set_title('empirical distribution', **kws_title)
  ax.text(-0.15, 1.02, 'D', transform=ax.transAxes, **kws_label)

  ax = axes[4]
  sns.heatmap(A, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title('transition matrix', **kws_title)
  ax.text(-0.15, 1.02, 'E', transform=ax.transAxes, **kws_label)

  ax = axes[5]
  ax.bar(i_arr, p2_arr)
  ax.set_xlabel('state number')
  ax.set_ylabel('probability')
  ax.set_title('predicted distribution', **kws_title)
  ax.text(-0.15, 1.02, 'F', transform=ax.transAxes, **kws_label)

  for ax in axes[[0,1,2,3,5]]:
    ax.tick_params(length=5, width=1)
    format_spine(ax)

  for ax in axes[[4]]:
    ax.tick_params(length=0)
    format_colorbar(ax)
    format_spine(ax)
    ax.set_xticks(i_arr-0.5+0.2)  # misalignment correction
    ax.set_yticks(i_arr-0.5+0.2)  # misalignment correction
    ax.set_xticklabels(i_arr, fontsize=10)
    ax.set_yticklabels(i_arr, fontsize=10)
    for i in (np.arange(n//4)*4)[1:]:
      ax.axvline(i, c='0.5', lw=1)
      ax.axhline(i, c='0.5', lw=1)

  fig.tight_layout(h_pad=2)
  fig.show()
  fig.savefig('tmp.png')
  return df

if __name__ == '__main__':
  hoge = plot_fig02()
