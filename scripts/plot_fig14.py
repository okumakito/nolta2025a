def plot_fig14():
  c1 = 5e-3   # regularization paramter 1
  c2 = 5e-3   # regularization paramter 2
  epsilon = 1e-10
  horizon = 4
  vmax = 0.2  # male: 0.35, female: 0.2
  gender = 'female'
  file_name = '../data/sc/data_hepato_female.csv'

  # load file
  df = pd.read_csv(file_name)
  df2 = df.groupby('month').clust.value_counts().unstack().T
  df2 = df2 / df2.sum()

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
  p0_arr = X[:,0]

  # emd
  out_list = []
  for i in range(X.shape[1]-1):
    C = ot.emd(X[:,i].copy(), X[:,i+1], D).T
    out_list.append(C)
  A = np.sum(out_list, axis=0)
  A += epsilon
  A /= A.sum(axis=0)
  n = len(A)

  # control
  r_arr = np.ones(n)
  r_arr[2] = -1
  #r_arr[8] = -1
  h_arr = np.array([0.5, 0.8, 0.9])
  Ap, F = calc_A_prime(A, r_arr, h_arr, c1=c1, c2=c2, p0_arr=p0_arr,
                       horizon=horizon)

  # simulation
  out_list = []
  for i in np.arange(X.shape[1]):
    out_list.append(np.linalg.matrix_power(A+Ap, i).dot(p0_arr))
  df4 = pd.DataFrame(out_list).T

  # plot
  fig, axes = plt.subplots(figsize=(10,5), ncols=2)
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_hmap = dict(cmap='Blues', xticklabels=1, yticklabels=1, vmin=0,
                  vmax=vmax,cbar_kws=dict(label='probability'))

  def draw_arrow(ax, Ap):
    for i in range(n):
      for j in range(n):
        if (i==j) or (Ap[i,j]==0):
          continue
        x1, y1 = c_df.loc[j]
        x2, y2 = c_df.loc[i]
        ax.arrow(x1, y1, x2-x1, y2-y1, color='C3', head_width=0.2,
                 length_includes_head=True)

  ax = axes[0]
  sns.scatterplot(data=df, x='x', y='y', hue='clust', ax=ax, s=1, ec='none',
                  palette='tab20', legend=False)
  xlim = ax.get_xlim()
  ax.axvspan(*xlim, color='w', alpha=0.5)
  sns.scatterplot(data=c_df, x='x', y='y', ax=ax, s=15, fc='0.2', ec='none',
                  legend=False, zorder=20)
  draw_arrow(ax, Ap)
  ax.tick_params(length=5, width=1)
  ax.set_aspect('equal')
  format_spine(ax)
  for idx, (x, y) in c_df.iterrows():
    ax.text(x, y, str(idx+1), c='k', ha='left', va='bottom')
  ax.set_xlabel('UMAP axis 1')
  ax.set_ylabel('UMAP axis 2')
  ax.set_title(f'interventions ({gender})', **kws_title)
  ax.text(-0.1, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.heatmap(data=df4, ax=ax, **kws_hmap)
  ax.tick_params(length=0)
  ax.set_xlabel('month')
  ax.set_ylabel('state')
  ax.set_xticklabels(df3.columns)
  ax.set_yticklabels(np.arange(20)+1)
  format_spine(ax)
  format_colorbar(ax)
  ax.set_title(f'simulation ({gender}, controlled)', **kws_title)
  ax.text(-0.1, 1.02, 'D', transform=ax.transAxes, **kws_label)
  
  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')

if __name__ == '__main__':
  plot_fig14()
