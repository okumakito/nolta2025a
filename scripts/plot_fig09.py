def plot_fig09():
  file_name = '../data/ndb/data_basic.csv'
  df = pd.read_csv(file_name)
  df['gender'] = df.gender.replace(dict(m='male', f='female'))
  model = PCA(n_components=2)
  X = model.fit_transform(z_score(df.iloc[:,4:-4]))
  df['x'] = X[:,0]
  df['y'] = X[:,1]
  perc1, perc2 = 100 * model.explained_variance_ratio_

  fig, axes = plt.subplots(figsize=(10,4), ncols=2)
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
  kws_legend = dict(bbox_to_anchor=(1,1), frameon=False, borderaxespad=0,
                    handletextpad=0, loc='upper left', markerscale=5)

  ax = axes[0]
  sns.scatterplot(data=df, x='x', y='y', hue='gender', s=5, ec='none',
                  palette='tab10', ax=ax)
  ax.legend(**kws_legend)
  ax.set_title('PCA (colored by gender)', **kws_title)
  ax.text(-0.15, 1.02, 'A', transform=ax.transAxes, **kws_label)

  ax = axes[1]
  sns.scatterplot(data=df, x='x', y='y', hue='age', s=5, ec='none',
                  palette='viridis', ax=ax)
  ax.legend(**kws_legend)
  handles, labels = ax.get_legend_handles_labels()
  labels = [x + '\u2013' + str(int(x)+4) for x in labels]
  ax.legend(handles, labels, **kws_legend)
  ax.set_title('PCA (colored by age)', **kws_title)
  ax.text(-0.15, 1.02, 'B', transform=ax.transAxes, **kws_label)

  for i, ax in enumerate(axes):
    ax.set_xlabel(f'PC1 ({perc1:.1f} %)')
    ax.set_ylabel(f'PC2 ({perc2:.1f} %)')
    ax.tick_params(length=5, width=1)
    format_spine(ax)

  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')
  return df

if __name__ == '__main__':
  hoge = plot_fig09()
