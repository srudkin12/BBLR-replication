#!/usr/bin/env python3
from __future__ import annotations
import hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source_data'; SRC.mkdir(exist_ok=True)
HALF=SRC/'elecdemand.csv'
DAILY=SRC/'elecdaily.csv'
URL='https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecdemand.csv'


def ensure_halfhourly():
    if not HALF.exists():
        print(f'Downloading canonical fpp2::elecdemand from {URL}')
        urllib.request.urlretrieve(URL, HALF)
    return HALF


def load_and_construct(write=True):
    ensure_halfhourly()
    h=pd.read_csv(HALF)
    need={'rownames','Demand','WorkDay','Temperature'}
    if not need.issubset(h.columns):
        raise RuntimeError(f'Half-hourly source missing columns: {sorted(need-set(h.columns))}')
    h=h[['rownames','Demand','WorkDay','Temperature']].copy()
    for c in h.columns: h[c]=pd.to_numeric(h[c],errors='coerce')
    if h.isna().any().any(): raise RuntimeError('NA/non-numeric values in canonical half-hourly source')
    h=h.sort_values('rownames').reset_index(drop=True)
    if len(h)!=17520: raise RuntimeError(f'Expected 17520 half-hourly rows, found {len(h)}')
    expected=np.arange(1,17521)
    if not np.array_equal(h.rownames.astype(int).to_numpy(),expected): raise RuntimeError('rownames are not exactly 1..17520')
    h['timestamp']=pd.Timestamp('2014-01-01 00:00:00')+pd.to_timedelta(h.rownames.astype(int)-1,unit='m')*30
    h['date']=h.timestamp.dt.normalize()
    h['slot']=h.timestamp.dt.strftime('%H:%M')
    counts=h.groupby('date').size()
    if len(counts)!=365 or not (counts==48).all(): raise RuntimeError('Not exactly 48 observations for each of 365 days')
    wu=h.groupby('date').WorkDay.nunique()
    if not (wu==1).all(): raise RuntimeError('WorkDay is not constant within every calendar day')
    ecount=h[h.slot=='18:00'].groupby('date').size()
    if len(ecount)!=365 or not (ecount==1).all(): raise RuntimeError('Not exactly one 18:00 observation per day')
    g=h.groupby('date',sort=True)
    daily=pd.DataFrame({
        'date':g.size().index,
        'Temperature':g.Temperature.max().to_numpy(),
        'WorkDay':g.WorkDay.first().to_numpy(),
        'PeakDemand':g.Demand.max().to_numpy(),
    })
    e=h[h.slot=='18:00'][['date','Demand','Temperature']].rename(columns={'Demand':'Demand1800','Temperature':'Temperature1800'})
    daily=daily.merge(e,on='date',how='left',validate='one_to_one')
    if daily.Demand1800.isna().any(): raise RuntimeError('Missing 18:00 demand after aggregation')
    # Cross-check daily temperature and workday against bundled canonical elecdaily.
    d=pd.read_csv(DAILY)
    need2={'rownames','Demand','WorkDay','Temperature'}
    if not need2.issubset(d.columns): raise RuntimeError('Bundled elecdaily source has unexpected columns')
    d=d[['rownames','WorkDay','Temperature']].copy()
    for c in d.columns: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.sort_values('rownames').reset_index(drop=True)
    if len(d)!=365: raise RuntimeError(f'Expected 365 daily rows, found {len(d)}')
    max_temp_err=float(np.max(np.abs(daily.Temperature.to_numpy()-d.Temperature.to_numpy())))
    wd_err=int(np.sum(daily.WorkDay.astype(int).to_numpy()!=d.WorkDay.astype(int).to_numpy()))
    if max_temp_err>1e-10: raise RuntimeError(f'Derived daily maximum temperature disagrees with elecdaily: max error {max_temp_err}')
    if wd_err!=0: raise RuntimeError(f'Derived WorkDay disagrees with elecdaily on {wd_err} days')
    iso=daily.date.dt.isocalendar()
    daily['week_group']=(iso['year'].astype(str)+'-W'+iso['week'].astype(str).str.zfill(2)).astype(str)
    daily['rownames']=np.arange(1,366)
    if write:
        daily.to_csv(SRC/'derived_peak_evening_daily.csv',index=False)
        audit=pd.DataFrame([{
            'half_hour_rows':len(h),'days':len(daily),'min_slots_per_day':int(counts.min()),'max_slots_per_day':int(counts.max()),
            'days_with_single_18_00':int(len(ecount)),'daily_temperature_max_abs_error':max_temp_err,'workday_mismatch_days':wd_err,
            'peak_min':float(daily.PeakDemand.min()),'peak_max':float(daily.PeakDemand.max()),
            'demand1800_min':float(daily.Demand1800.min()),'demand1800_max':float(daily.Demand1800.max()),
            'temp_min':float(daily.Temperature.min()),'temp_max':float(daily.Temperature.max()),
            'workdays':int((daily.WorkDay==1).sum())
        }])
        audit.to_csv(SRC/'construction_audit.csv',index=False)
    return h,daily,max_temp_err,wd_err


def main():
    h,d,e,w=load_and_construct(write=True)
    print('PEAK_EVENING_PREFLIGHT')
    print(f'half_hour_rows={len(h)} days={len(d)} workdays={(d.WorkDay==1).sum()}')
    print(f'temperature_crosscheck_max_abs_error={e:.3g} workday_mismatch_days={w}')
    print(f'peak_range={d.PeakDemand.min():.4g}..{d.PeakDemand.max():.4g} GW')
    print(f'demand_18_00_range={d.Demand1800.min():.4g}..{d.Demand1800.max():.4g} GW')
    print('PASS_FOR_PEAK_EVENING_PRODUCTION')

if __name__=='__main__': main()
