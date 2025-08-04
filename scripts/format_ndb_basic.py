def format_ndb_basic():
  data_file = '../data/ndb/001258742_basic.csv'
  rename_file = '../data/ndb/rename_list.csv'
  df = pd.read_csv(data_file, skiprows=5, header=None)
  sr = pd.read_csv(rename_file, index_col=0).squeeze()
  df.columns = ['pref', 'area_code', 'area_name', 'var_name',
                'm40', 'm45', 'm50', 'm55', 'm60', 'm65', 'm70', 'mall',
                'f40', 'f45', 'f50', 'f55', 'f60', 'f65', 'f70', 'fall']
  df = df.ffill()
  df['area_name'] = df.pref + '_' + df.area_name
  df['area_code'] = df.area_code.astype(int)
  df['var_name'] = df.var_name.replace(sr)
  df = df[~df.pref.isin(['二次医療圏不明','全国'])]
  df = df.drop(['pref','mall','fall'], axis=1)

  out_list = []
  for area_code, sub_df in df.groupby('area_code'):
    area_name = sub_df['area_name'].values[0]
    df2 = sub_df.drop(['area_code','area_name'], axis=1)
    df2 = df2.set_index('var_name').T
    df2.insert(0, 'area_code', area_code)
    df2.insert(1, 'area_name', area_name)
    df2.insert(2, 'gender', df2.index.str[0])
    df2.insert(3, 'age', df2.index.str[1:].astype(int))
    out_list.append(df2)
  df = pd.concat(out_list, axis=0)

  df.to_csv('tmp.csv', encoding='utf-8-sig', index=False)
  return df

if __name__ == '__main__':
  hoge = format_ndb_basic()
