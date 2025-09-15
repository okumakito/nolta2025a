def plot_fig13a():
  file_name1 = '../data/sc/data_hepato_male.csv'
  file_name2 = '../data/sc/data_hepato_female.csv'
  df1 = pd.read_csv(file_name1)
  df2 = pd.read_csv(file_name2)
  month_arr = np.sort(df2.month.unique())

  fig, axes = plt.subplots(figsize=(12,6), nrows=2, ncols=5)
  axes = axes.flatten()
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')

  for i in range(5):
    month = month_arr[i]
    ax = axes[i]
    sns.scatterplot(data=df1[df1.month==month], ax=ax, x='x', y='y',
                    s=1, ec='none', color=plt.cm.tab10(i), alpha=0.2)
    ax.set_title(f'\u2642 {month} months', **kws_title)

  for i in range(5):
    month = month_arr[i]
    ax = axes[i+5]
    sns.scatterplot(data=df2[df2.month==month], ax=ax, x='x', y='y',
                    s=1, ec='none', color=plt.cm.tab10(i), alpha=0.2)
    ax.set_title(f'\u2640 {month} months', **kws_title)

  ax = axes[0]
  ax.text(-0.05, 1.02, 'A', transform=ax.transAxes, **kws_label)
  ax = axes[5]
  ax.text(-0.05, 1.02, 'B', transform=ax.transAxes, **kws_label)

  for ax in axes:
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP axis 1')
    ax.set_ylabel('UMAP axis 2')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
      spine.set_linewidth(1)

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return df1

if __name__ == '__main__':
  hoge = plot_fig13a()
