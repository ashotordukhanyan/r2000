import pandas as pd
r2 = pd.read_csv('data/kaggle/r2.csv')
r2['date'] =  pd.to_datetime(r2['date'])
r2['dvolume'] = r2['volume'] * r2['close']
#r2.set_index(['Name','date'],inplace=True, verify_integrity=True,drop=False)
offset_pre = -10
offset_post = 10

def label(prefix,offset):
    suffix = '%s_T%s%d' % (prefix,"plus" if offset > 0 else 'minus', abs(offset))

def enrich_security(df):
    df2 = df.copy()
    r2['dvolume'] = r2['volume'] * r2['close']
    df2['ADV']= df2['dvolume'].rolling(21).median()
    for offset in range(offset_pre,offset_post) :
        if offset != 0:
            df2[label('close',offset)]=df2['close'].shift(offset)
    return df2
def calc_returns(df):
    for offset in range(offset_pre, offset_post):
        if offset != 0:
            suffix = '_T%s%d' % ("plus" if offset > 0 else 'minus', offset)
            if ( offset < 0 ) :
                df[label('r',offset)] = (df['close'] - df[label('close',offset)] ) / df['close']
            else:
                df[label('r',offset)] = (df[label('close',offset)] - df['close']) / df['close']

enriched = {}
groups = r2.groupby('Name', as_index=False)
for name,group in groups:
    print('Enriching ', name )
    enriched[name] = enrich_security(group)

r2 = pd.concat(enriched.values())
calc_returns(r2)
##pd.to_pickle(r2enriched,'./data/r2enricked.pkl')
#r2['dvolume']=r2['volume']*r2['close']
mvolume = r2.groupby('date',as_index=False).dvolume.sum()
mvolume = mvolume.rename(index=str, columns={"dvolume": "mkt_volume"})
mvolume['AMV']= mvolume['mkt_volume'].rolling(21).median()
r2=pd.merge(r2,mvolume,on='date',how='outer')
