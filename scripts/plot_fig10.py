def plot_fig10():
  n  = 10     # number of states
  c1 = 0.001  # regularization parameter 1
  c2 = 0.001  # regularization parameter 2
  epsilon = 1e-10

  # load data
  file_name = '../data/ndb/data_basic.csv'
  df = pd.read_csv(file_name)
  df = df[df.gender=='m'].iloc[:,4:-4].copy()
  v_df = z_score(df)
  n_area = len(v_df)//7  # 335

  # discretized time series
  model = KMeans(n_clusters=n, random_state=0, n_init='auto')
  model.fit(v_df)
  v_df['state'] = model.predict(v_df)
  v_df['time'] = np.tile(np.arange(7), n_area)
  v_df['area'] = np.repeat(np.arange(n_area),7)

  # reset cluster number in the order of age
  sr = v_df.groupby('state').time.mean().argsort().argsort()
  v_df['state'] = v_df.state.replace(sr) + 1

  # cluster centers
  df['state'] = v_df.state.values
  c_df = df.groupby('state').mean()
  cz_df = z_score(c_df)

  # state-time matrix
  st_df = v_df[['state','time']].value_counts().unstack().\
    fillna(0).astype(int)
  p_arr = st_df.sum(axis=1)
  p_arr = p_arr / p_arr.sum()

  # transition matrix, without reset
  df2 = pd.DataFrame()
  df2['src'] = v_df[v_df.time!=6].state.values
  df2['dst'] = v_df[v_df.time!=0].state.values
  df2 = df2.value_counts().unstack().fillna(0).T
  n_df = df2.astype(int).copy()
  df2 /= df2.sum()

  # transition matrix, with reset
  df3 = pd.DataFrame()
  df3['src'] = v_df.state.values
  df3['dst'] = v_df.groupby('area').state.\
    apply(lambda x:pd.Series(np.roll(x,-1))).values
  df3 = df3.value_counts().unstack().fillna(0).T
  A = df3.values + epsilon
  A = A / A.sum(axis=0)

  # control
  r_arr = np.ones(n)
  r_arr[-1] = -1
  h_arr = np.array([0.5])
  Ap, F = calc_A_prime(A, r_arr, h_arr, c1=c1, c2=c2)
  p2_arr = calc_stationary(A+Ap)

  # simulation, without resset
  A2 = df2.values + Ap
  p0_arr = st_df[0].values
  p0_arr = p0_arr / p0_arr.sum()
  out_list = []
  for i in np.arange(7):
    out_list.append(np.linalg.matrix_power(A2, i).dot(p0_arr))
  fr_df = pd.DataFrame(out_list).T

  fig = plt.figure(figsize=(12,12))
  gs = fig.add_gridspec(3,2)
  ax1 = fig.add_subplot(gs[0,:])
  ax2 = fig.add_subplot(gs[1,0])
  ax3 = fig.add_subplot(gs[1,1])
  ax4 = fig.add_subplot(gs[2,0])
  ax5 = fig.add_subplot(gs[2,1])
  axes = np.array([ax1, ax2, ax3, ax4, ax5])
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_cbar = dict(cmap='Blues', label='probability', aspect=10)

  ax = axes[0]
  sns.heatmap(data=cz_df, ax=ax, cmap='RdBu_r', annot=c_df, fmt='.1f',
              cbar_kws=dict(label='z score', aspect=10, pad=0.03))
  labels = [x.get_text().upper().replace('HBA1C','HbA1c').\
            replace('GGTP','$\gamma$-GTP') for x in ax.get_xticklabels()]
  ax.set_xticklabels(labels)
  ax.set_title('center of each state', **kws_title)
  ax.text(-0.04, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.heatmap(data=st_df/st_df.sum(), ax=ax, cmap='Blues', annot=st_df,
              fmt='d', cbar_kws=kws_cbar, vmax=1)
  format_spine(ax)
  ax.set_xlabel('age group')
  ax.set_xticks(np.arange(7)+0.5, np.arange(40,75,5))
  ax.set_title('state distribution by age group', **kws_title)
  ax.text(-0.09, 1.02, 'B', transform=ax.transAxes, **kws_label)

  ax = axes[2]
  sns.heatmap(data=df2, ax=ax, cmap='Blues', annot=n_df, fmt='d',
              cbar_kws=kws_cbar, vmax=1)
  format_spine(ax)
  ax.set_xlabel('source')
  ax.set_ylabel('destination')
  for i in range(n):
    for j in range(n):
      if (i==j) or (Ap[i,j]==0):
        continue
      ax.plot([j,j+1,j+1,j,j], [i,i,i+1,i+1,i], lw=2, c='C3')
  ax.set_title('transition matrix without reset', **kws_title)
  ax.text(-0.09, 1.02, 'C', transform=ax.transAxes, **kws_label)

  ax = axes[3]
  df_tmp = pd.DataFrame()
  df_tmp['state'] = np.tile(np.arange(n),2)
  df_tmp['prob'] = np.hstack([p_arr, p2_arr])
  df_tmp['cond'] = ['original']*n + ['controlled']*n
  sns.barplot(data=df_tmp, x='state', y='prob', hue='cond', ax=ax)
  ax.tick_params(length=5, width=1)
  format_spine(ax)
  ax.set_ylabel('probability')
  ax.set_xticks(np.arange(n), np.arange(n)+1)
  ax.legend(frameon=False, loc='upper center', ncols=2)
  ax.set_xlim((-1,n))
  ax.set_ylim((0,0.35))
  ax.set_title('predicted distribution', **kws_title)
  ax.text(-0.09, 1.02, 'D', transform=ax.transAxes, **kws_label)

  ax = axes[4]
  sns.heatmap(data=fr_df, ax=ax, cmap='Blues', cbar_kws=kws_cbar,
              vmax=1)
  format_spine(ax)
  ax.set_xlabel('age group')
  ax.set_ylabel('state')
  ax.set_xticks(np.arange(7)+0.5, np.arange(40,75,5))
  ax.set_yticks(np.arange(n)+0.5, np.arange(n)+1)
  ax.set_title('simulation (controlled)', **kws_title)
  ax.text(-0.09, 1.02, 'E', transform=ax.transAxes, **kws_label)

  for ax in axes[[0,1,2,4]]:
    ax.tick_params(length=0)
    format_colorbar(ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

  fig.tight_layout(h_pad=2)
  fig.show()
  fig.savefig('tmp.png')
  return v_df

if __name__ == '__main__':
  hoge = plot_fig10()
