"""
Tests for toolkit.data.ninetyfiveofive.load_9505_stencil_pool and its site-mapping helper.

Uses small hand-built xarray Datasets (written to tmp_path as NetCDF) rather than the real
9505 master files, so these tests don't depend on data being present on disk.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from toolkit.data.ninetyfiveofive import _map_sites_to_reach_vars, load_9505_stencil_pool


def _pcp_reach_mapping():
    return pd.DataFrame({
        "PCP_NAME": ["INK20000", "IN8BEMA", "INBARP"],
        "REACH_COMID": [111, 222, 333],
    })


def test_map_sites_to_reach_vars_whitespace_insensitive():
    mapping = _pcp_reach_mapping()
    # Colorado (unpadded), Trinity (1 embedded space), Sabine (2 embedded spaces)
    site_names = ["INK20000", "IN 8BEMA", "IN  BARP"]

    result = _map_sites_to_reach_vars(site_names, mapping)

    assert result == ["reach_111", "reach_222", "reach_333"]


def test_map_sites_to_reach_vars_missing_site_raises():
    mapping = _pcp_reach_mapping()

    with pytest.raises(ValueError, match="NOT_A_SITE"):
        _map_sites_to_reach_vars(["INK20000", "NOT_A_SITE"], mapping)


def _write_period_nc(path, ensemble_filenames, reach_data, n_years):
    """Build a minimal 9505-shaped NetCDF: dims (ensemble_id, time_mn), reach_<comid>
    variables, plus the 1D metadata variables filter_ensemble_members/create_metadata_df need.

    reach_data : dict[int, np.ndarray] mapping COMID -> array of shape (n_members, n_years*12)
    """
    n_members = len(ensemble_filenames)
    time_mn = pd.date_range("2020-01-01", periods=n_years * 12, freq="MS")

    data_vars = {
        f"reach_{comid}": (["ensemble_id", "time_mn"], values)
        for comid, values in reach_data.items()
    }
    data_vars["original_filename"] = (["ensemble_id"], ensemble_filenames)

    ds = xr.Dataset(
        data_vars,
        coords={"ensemble_id": ensemble_filenames, "time_mn": time_mn},
    )
    ds.to_netcdf(path)


def test_load_9505_stencil_pool_shape_and_column_order(tmp_path):
    mapping = _pcp_reach_mapping()
    site_names = ["INK20000", "IN 8BEMA"]  # deliberately reversed vs. reach insertion order below
    n_members, n_years = 2, 3

    reach_data = {
        111: np.arange(n_members * n_years * 12).reshape(n_members, n_years * 12).astype(float),
        222: -np.arange(n_members * n_years * 12).reshape(n_members, n_years * 12).astype(float),
    }
    ensemble_filenames = [
        "PRMS_RAPID_BCC-CSM2-MR_ssp245_r1i1p1f1_DBCCA_Daymet_2020_2059.nc",
        "VIC5_RAPID_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Livneh_2020_2059.nc",
    ]
    nc_path = tmp_path / "master_streamflow_2020_2059_af.nc"
    _write_period_nc(nc_path, ensemble_filenames, reach_data, n_years)

    result = load_9505_stencil_pool(
        site_names=site_names,
        pcp_reach_mapping=mapping,
        nc_paths={"2020_2059": nc_path},
        periods=["2020_2059"],
    )

    assert result.shape == (n_members * n_years * 12, len(site_names))
    # Column 0 (INK20000 -> reach_111) must be the positive series, column 1 (IN 8BEMA ->
    # reach_222) the negated one -- proves column order follows site_names, not reach insertion.
    assert (result[:, 0] >= 0).all()
    assert (result[:, 1] <= 0).all()


def test_load_9505_stencil_pool_pools_multiple_periods(tmp_path):
    mapping = _pcp_reach_mapping()
    site_names = ["INK20000"]
    n_members, n_years = 1, 2
    ensemble_filenames = ["PRMS_RAPID_BCC-CSM2-MR_ssp245_r1i1p1f1_DBCCA_Daymet_2020_2059.nc"]

    reach_data = {111: np.ones((n_members, n_years * 12))}
    path_a = tmp_path / "a.nc"
    path_b = tmp_path / "b.nc"
    _write_period_nc(path_a, ensemble_filenames, reach_data, n_years)
    _write_period_nc(path_b, ensemble_filenames, reach_data, n_years)

    single = load_9505_stencil_pool(
        site_names=site_names,
        pcp_reach_mapping=mapping,
        nc_paths={"a": path_a, "b": path_b},
        periods=["a"],
    )
    pooled = load_9505_stencil_pool(
        site_names=site_names,
        pcp_reach_mapping=mapping,
        nc_paths={"a": path_a, "b": path_b},
        periods=["a", "b"],
    )

    assert pooled.shape[0] == 2 * single.shape[0]
    assert pooled.shape[1] == single.shape[1]


def test_load_9505_stencil_pool_applies_ensemble_filters(tmp_path):
    mapping = _pcp_reach_mapping()
    site_names = ["INK20000"]
    n_years = 1
    ensemble_filenames = [
        "PRMS_RAPID_BCC-CSM2-MR_ssp245_r1i1p1f1_DBCCA_Daymet_2020_2059.nc",
        "VIC5_RAPID_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Livneh_2020_2059.nc",
    ]
    reach_data = {111: np.array([[1.0] * 12, [2.0] * 12])}
    nc_path = tmp_path / "master_streamflow_2020_2059_af.nc"
    _write_period_nc(nc_path, ensemble_filenames, reach_data, n_years)

    filtered = load_9505_stencil_pool(
        site_names=site_names,
        pcp_reach_mapping=mapping,
        nc_paths={"2020_2059": nc_path},
        periods=["2020_2059"],
        ensemble_filters={"hydro_model": ["PRMS"]},
    )

    # Only the PRMS member (value 1.0) should survive the filter.
    assert filtered.shape == (12, 1)
    np.testing.assert_allclose(filtered[:, 0], 1.0)
