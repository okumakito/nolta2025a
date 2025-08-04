def plot_fig12b():
  file_name = '../data/sc/data_hepato_male.csv'
  df = pd.read_csv(file_name)
  df['clust'] += 1
  df2 = df.groupby('clust', as_index=False).mean()

  fig, ax = plt.subplots(figsize=(6,6))
  kws_title = dict(fontsize=16, pad=10)
  kws_label = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')

  sns.scatterplot(data=df, x='x', y='y', hue='clust', ax=ax, s=1, ec='none',
                  palette='tab20')
  sns.scatterplot(data=df2, x='x', y='y', ax=ax, s=15, fc='0.2', ec='none',
                  legend=False)
  ax.legend(bbox_to_anchor=(1,1), loc='upper left', frameon=False,
            borderaxespad=0, handletextpad=0, labelspacing=0.2,
            markerscale=10)
  ax.tick_params(length=5, width=1)
  ax.set_aspect('equal')
  format_spine(ax)
  ax.set_xlabel('UMAP axis 1')
  ax.set_ylabel('UMAP axis 2')
  ax.set_title('K-means partition (male)', **kws_title)
  ax.text(-0.1, 1.02, 'C', transform=ax.transAxes, **kws_label)
  
  fig.tight_layout()
  fig.show()
  fig.savefig('tmp.png')

if __name__ == '__main__':
  plot_fig12b()
