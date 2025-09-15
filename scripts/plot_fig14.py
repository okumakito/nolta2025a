def plot_fig14():
  epsilon = 1e-10
  file_name = '../data/sc/data_hepato_male.csv'
  gender = 'male'
  vmax = 0.35  # male: 0.35, female: 0.2
  df = pd.read_csv(file_name)
  df2 = df.groupby('month').clust.value_counts().unstack().T
  df2 = df2 / df2.sum()
  print(df2.max().max())

  # interpolation
  df3 = df2.copy()
  df3[8] = (2*df2[6] + df2[12])/3
  df3[13] = (3*df2[12] + df2[16])/4
  df3[18] = (5*df2[16] + 2*df2[23])/7
  df3 = df3.drop([6,12,16], axis=1)
  df3 = df3.sort_index(axis=1)

  # smoothing
  df3 = df3.rolling(3, min_periods=1, center=True, axis=1).mean()
  df3 = df3 / df3.sum()

  # distance matrix
  X = df3.values
  c_df = df.groupby('clust').mean().drop('month', axis=1)
  D = distance.squareform(distance.pdist(c_df))

  # emd
  out_list = []
  for i in range(X.shape[1]-1):
    C = ot.emd(X[:,i].copy(), X[:,i+1], D).T
    out_list.append(C)
  A = np.sum(out_list, axis=0)
  A += epsilon
  A /= A.sum(axis=0)

  # simulation
  p0_arr = X[:,0]
  out_list = []
  for i in np.arange(X.shape[1]):
    out_list.append(np.linalg.matrix_power(A, i).dot(p0_arr))
  df4 = pd.DataFrame(out_list).T

  fig, axes = plt.subplots(figsize=(12,6), ncols=3)
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_hmap = dict(cmap='Blues', xticklabels=1, yticklabels=1, vmin=0,
                  vmax=vmax, cbar_kws=dict(label='probability',
                                           orientation='horizontal'))

  ax = axes[0]
  sns.heatmap(data=df2, ax=ax, **kws_hmap)
  ax.set_title(f'original time series\n({gender})', **kws_title)
  ax.text(-0.1, 1.07, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.heatmap(data=df3, ax=ax, **kws_hmap)
  ax.set_title(f'interpolation and smoothing\n({gender})', **kws_title)
  ax.text(-0.1, 1.07, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  sns.heatmap(data=df4, ax=ax, **kws_hmap)
  ax.set_xticklabels(df3.columns)
  ax.set_title(f'simulation\n({gender})', **kws_title)
  ax.text(-0.1, 1.07, 'C', transform=ax.transAxes, **kws_label)

  for ax in axes:
    ax.tick_params(length=0)
    ax.set_xlabel('month')
    ax.set_ylabel('state')
    ax.set_yticklabels(np.arange(20)+1)
    format_spine(ax)
    format_colorbar(ax)

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return df2

if __name__ == '__main__':
  hoge = plot_fig14()
