import pandas as pd
from pathlib import Path
from bblr.univariate import DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY
from bblr.repeated_cv import selector_rows

def test_rcv1000_panel_is_unique_and_broad():
    root=Path(__file__).resolve().parents[1]
    d=pd.read_csv(root/'RCV1000_OPERATING_CELLS.csv')
    assert len(d)==92
    assert d.dgp.nunique()>=30
    assert d.panel_block.nunique()==8
    assert not d.duplicated(['dgp','design','n_train','noise_sd']).any()
    assert set(d.dgp).issubset(DGP_REGISTRY)
    allowed=set(SUPPORT_DESIGN_REGISTRY)|{'uniform','normal','skewed','gap'}
    assert set(d.design).issubset(allowed)

def test_rcv1000_panel_contains_good_and_bad_mechanism_stresses():
    root=Path(__file__).resolve().parents[1]
    d=pd.read_csv(root/'RCV1000_OPERATING_CELLS.csv')
    assert {'A_shape_smooth','B_nonsmooth','C_density_geometry','D_support_ladder','E_width_support','F_location_gap','G_noise_scaling','H_sample_scaling'}==set(d.panel_block)
    assert {'F07','B06','R04C03','U01'}.issubset(set(d.dgp))
    assert {'D10','G60','skewed','normal','gap'}.issubset(set(d.design))

def test_selector_rows_retains_llr_flags():
    import pandas as pd
    g=pd.DataFrame({
      'test_slope_rmse':[1.0,2.0], 'oracle_slope':[True,False],
      'selected_repeated_cv':[True,False], 'llr_selected_repeated_cv':[False,True],
      'r':[.2,.3], 'eps':[.1,0.0], 'test_y_rmse':[1.0,1.0]
    })
    s=selector_rows(g)
    assert 'llr_selected_repeated_cv' in set(s.selector)
