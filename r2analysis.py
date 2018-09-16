from datetime import datetime
from scipy.stats.stats import pearsonr

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

def label(prefix, offset):
    return '%s_T%s%d' % (prefix, "plus" if offset > 0 else 'minus', abs(offset))


def get_dataset():
    r2 = pd.read_csv('data/kaggle/r2.csv')
    r2['date'] =  pd.to_datetime(r2['date'])
    r2['dvolume'] = r2['volume'] * r2['close']
    r2['dvol'] = (r2['high']  - r2['low'])/r2['open']

    #r2.set_index(['Name','date'],inplace=True, verify_integrity=True,drop=False)
    offset_pre = -5
    offset_post = 5

    def enrich_security(df):
        df2 = df.copy()
        df2['ADV']= df2['dvolume'].rolling(21).median()
        df2['EDV'] = df2['dvolume']/df2['ADV']
        for offset in range(offset_pre, 0):
            df2[label('EDV', offset)] = df2['EDV'].shift(offset)

        df2['ADVOL'] = df2['dvol'].rolling(21).median()
        df2['EDVOL'] = df2['dvol'] / df2['ADVOL']

        for offset in range(offset_pre, 0):
            df2[label('EDVOL', offset)] = df2['EDVOL'].shift(offset)

        for offset in range(offset_pre,offset_post) :
            if offset != 0:
                df2[label('close',offset)]=df2['close'].shift(offset)

        return df2

    def calc_returns(df):
        for offset in range(offset_pre, offset_post):
            if offset != 0:
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
    mvolume['EMV']= mvolume['mkt_volume']/mvolume['AMV'] ##Market excess volume
    for offset in range(offset_pre,0):
        mvolume[label('EMV',offset)] = mvolume['EMV'].shift(offset)

    r2=pd.merge(r2,mvolume,on='date',how='outer')
    return r2

def run_regression(year,data,features):
    print("Runing for year ", year)
    df = data[(data['date'] >= datetime(year, 1, 1)) & (data['date'] < datetime(year+1, 1, 1)) ]
    train_x, test_x,train_y,test_y = train_test_split(df[features], df['r_Tplus1'], test_size=0.4, random_state=42)
    rf = RandomForestRegressor(random_state=42, verbose=100, n_estimators=50, n_jobs=10)
    print("Fitting a dataset containing %d observations and %d features"%(len(train_x), len(train_x.columns)))
    rf.fit(train_x,train_y)
    print("Done fitting")
    predict_y = rf.predict(test_x)
    result = {}
    result['importance'] = dict(zip(features,rf.feature_importances_))
    result['corr'] = pearsonr(test_y,predict_y)
    return result

if __name__ == '__main__':
    try:
        r2 = pd.read_pickle('./data/r2.pkl')
    except: r2 = None
    if r2 is  None or len(r2)==0:
        r2 = get_dataset()
        pd.to_pickle(r2,'./data/r2.pkl')

    r2.replace([np.inf, -np.inf], np.nan, inplace = True)
    r2.dropna( inplace = True)
    features = []

    for t in ['r', 'EMV', 'EDV', 'EDVOL']:
        features += [label(t, offset) for offset in range(-5, 0)]

    stats = {}
    for d in range(2014,2019):
       stats[d] = run_regression(d,r2, features)

    results = pd.DataFrame(columns=['year','correlation'] + features )
    years = list(stats.keys())
    results['year'] = years
    results['correlation'] = [ stats[y]['corr'] for y in years ]
    for f in features:
        results['correlation'] = [stats[y]['corr'] for y in years]
        results[f] = [ stats[y]['importance'][f] for y in years]
    print(results)
    pd.to_pickle(results, './data/results.pkl')