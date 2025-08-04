def plot_fig03():
  n      = 20        # number of states
  n_step = 10**3     # number of time steps
  d      = 1.5       # data range
  dt     = 0.01      # delta t
  sigma  = 1.0       # noise intensity
  epsilon = 1e-10
  gamma1 = 20
  gamma2 = 5
  np.random.seed(123)

  # time series
  x = 0
  sigma2  = sigma * dt**0.5
  def f(x):
    return -4 * x**3 +4 * x
  out_list = []
  for _ in range(n_step):
    x = x + f(x) * dt + sigma2 * np.random.randn()
    out_list.append(x)
  x_arr = np.array(out_list)

  # discretized time series
  xd_arr = (n * (x_arr + d) / (2*d)).astype(int)
  xd_arr = np.clip(xd_arr, 0, n-1) + 1

  # transition matrix
  df = pd.DataFrame()
  df['src'] = xd_arr[:-1]
  df['dst'] = xd_arr[1:]
  df = df.value_counts().unstack().fillna(0).T
  df /= df.sum()
  A = df.values
  A += epsilon
  A /= A.sum(axis=0)

  # smoothing
  xx_arr = np.linspace(-d, d, n)
  D = np.abs(xx_arr.reshape(-1,1) - xx_arr)
  A2 = calc_smoothing(A, D, gamma1)
  A3 = calc_smoothing(A, D, gamma2)

  # plot
  fig, axes = plt.subplots(figsize=(12,4), ncols=3)
  kws_title = dict(fontsize=16, pad=10)
  kws_hmap  = dict(cmap='Blues', xticklabels=1, yticklabels=1, vmax=0.7,
                   vmin=0)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  i_arr = np.arange(n)+1

  ax = axes[0]
  sns.heatmap(A, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title('transition matrix\nwithout smoothing', **kws_title)
  ax.text(-0.15, 1.1, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.heatmap(A2, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title(f'weak smoothing\n($\gamma={gamma1}$)', **kws_title)
  ax.text(-0.15, 1.1, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  sns.heatmap(A3, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title(f'strong smoothing\n($\gamma={gamma2}$)', **kws_title)
  ax.text(-0.15, 1.1, 'C', transform=ax.transAxes, **kws_label)

  for ax in axes:
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

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return df

if __name__ == '__main__':
  hoge = plot_fig03()
