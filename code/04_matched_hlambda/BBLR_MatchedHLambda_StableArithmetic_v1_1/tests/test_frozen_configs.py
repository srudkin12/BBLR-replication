from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_config_shape():
    d=pd.read_csv(ROOT/'data/FROZEN_SELECTED_CONFIGS.csv.gz');assert len(d)==92000;assert d.cell_id.nunique()==92;assert d.groupby('cell_id').size().min()==1000;assert d.groupby('cell_id').size().max()==1000
