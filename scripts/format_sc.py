def format_sc():
  #file_name = '../data/sc/GSM7899299_PanSci_gWAT_df_cell.csv'
  file_name = '../data/sc/GSM7899274_PanSci_liver_df_cell.csv'
  df = pd.read_csv(file_name)

  #df = df[(df.gender == 'Male') & (df.genotype == 'WT') &
  #        (df.main_cell_type_organ == 'Adipocytes-gWAT')]
  df = df[(df.gender == 'Male') & (df.genotype == 'WT') &
          (df.main_cell_type_organ == 'Hepatocytes-Liver')]
  df['age'] = df.age_group.str[:2].astype(int)
  df = df[['umap_1', 'umap_2', 'age']]
  df.columns = ['x', 'y', 'month']

  # clustering
  n = 20
  model = KMeans(n_clusters=n, random_state=0)
  model.fit(df[['x','y']])
  df['clust'] = model.predict(df[['x','y']])
  Z = linkage(model.cluster_centers_, metric='euclidean', method='ward')
  sr = pd.Series(np.arange(n), index=leaves_list(Z))
  df['clust'] = df.clust.replace(sr)

  df.to_csv('tmp.csv', index=False)
  return df

if __name__ == '__main__':
  hoge = format_sc()
