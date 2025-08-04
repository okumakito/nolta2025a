def plot_fig04():
  n      = 20        # number of states
  n_step = 10**4     # number of time steps
  d      = 1.7       # data range
  dt     = 0.01      # delta t
  sigma  = 1.0       # noise intensity
  epsilon = 1e-10
  c_list = [(0.1,0.1), (0.05,0.05), (0.01,0.01)]  # regularization
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

  # control
  r_arr = np.ones(n)
  r_arr[n//2:] = -1
  h_arr = np.array([0.5, 0.8, 0.9])
  out_list = []
  for (c1,c2) in c_list:
    Ap, F = calc_A_prime(A, r_arr, h_arr, c1=c1, c2=c2)
    out_list.append(Ap)
  Ap1, Ap2, Ap3 = out_list
  p1_arr = calc_stationary(A+Ap1)
  p2_arr = calc_stationary(A+Ap2)
  p3_arr = calc_stationary(A+Ap3)

  # plot
  fig, axes = plt.subplots(figsize=(12,8), ncols=3, nrows=2)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_hmap  = dict(cmap='RdBu_r', xticklabels=1, yticklabels=1, vmax=0.3,
                   vmin=-0.3)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  i_arr = np.arange(n)+1

  txt1 = f"$(\lambda_1=\lambda_2={c_list[0][0]:.2f})$"
  txt2 = f"$(\lambda_1=\lambda_2={c_list[1][0]:.2f})$"
  txt3 = f"$(\lambda_1=\lambda_2={c_list[2][0]:.2f})$"
  
  ax = axes[0]
  sns.heatmap(Ap1, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title(f"$A'$ " + txt1, **kws_title)
  ax.text(-0.15, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.heatmap(Ap2, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title(f"$A'$ " + txt2, **kws_title)
  ax.text(-0.15, 1.02, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  sns.heatmap(Ap3, ax=ax, **kws_hmap)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  ax.set_title(f"$A'$ " + txt3, **kws_title)
  ax.text(-0.15, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[3]
  ax.bar(i_arr, p1_arr)
  ax.set_xlabel('state number')
  ax.set_ylabel('probability')
  ax.set_title('predicted distribution\n' + txt1, **kws_title)
  ax.text(-0.15, 1.1, 'D', transform=ax.transAxes, **kws_label)

  ax = axes[4]
  ax.bar(i_arr, p2_arr)
  ax.set_xlabel('state number')
  ax.set_ylabel('probability')
  ax.set_title('predicted distribution\n' + txt2, **kws_title)
  ax.text(-0.15, 1.1, 'E', transform=ax.transAxes, **kws_label)

  ax = axes[5]
  ax.bar(i_arr, p3_arr)
  ax.set_xlabel('state number')
  ax.set_ylabel('probability')
  ax.set_title('predicted distribution\n' + txt3, **kws_title)
  ax.text(-0.15, 1.1, 'F', transform=ax.transAxes, **kws_label)

  for ax in axes[:3]:
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

  for ax in axes[3:]:
    ax.tick_params(length=5, width=1)
    format_spine(ax)
    ax.set_ylim((0,0.3))

  fig.tight_layout(h_pad=2)
  fig.show()
  fig.savefig('tmp.png')
  return df

if __name__ == '__main__':
  hoge = plot_fig04()
