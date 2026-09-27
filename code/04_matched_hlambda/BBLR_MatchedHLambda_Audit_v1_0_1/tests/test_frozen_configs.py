from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_archive_shape():
    x=pd.read_csv(ROOT/'data/FROZEN_SELECTED_CONFIGS.csv.gz')
    assert len(x)==92000
    assert x.cell_id.nunique()==92
    sizes=x.groupby('cell_id').size()
    assert sizes.min()==1000 and sizes.max()==1000
    assert (x.r>0).all() and (x.eps>=0).all()
