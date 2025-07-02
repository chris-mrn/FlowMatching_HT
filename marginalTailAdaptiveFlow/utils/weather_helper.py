""" Mostly taken from https://zenodo.org/record/5081927#.YphTbXVBxhE """
from typing import Tuple, NamedTuple, List, Optional, Union
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.lines import Line2D
from pathlib import Path

USE_CLOUD_OPTICAL_DEPTH = True

class StackInfoVar(NamedTuple):
    name: str
    dims: Tuple[str]
    shape: Tuple[int]

StackInfo = List[StackInfoVar]

# Specialized stacking/unstacking functions (as opposed to
# using xarray's to_stacked_array/to_unstacked_dataset).
# This allows to have control over the exact stacking behaviour
# which in turn allows to store compact stacking metadata and use it
# to unstack arbitrary arrays not directly related to the input dataset object.
def to_stacked_array(ds: xr.Dataset, var_names=None, new_dim='stacked', name=None) -> Tuple[xr.DataArray, StackInfo]:
    # Sample dimension must be the first dimension in all variables.
    if not var_names:
        var_names = sorted(ds.data_vars)
    stack_info = []
    var_stacked = []
    for var_name in var_names:
        v = ds.data_vars[var_name]
        if len(v.dims) > 1:
            stacked = v.stack({new_dim: v.dims[1:]})
            stacked = stacked.drop(list(stacked.coords.keys()))
        else:
            stacked = v.expand_dims(new_dim, axis=-1)
        stack_info.append(StackInfoVar(var_name, v.dims, v.shape[1:]))
        var_stacked.append(stacked)
    arr = xr.concat(var_stacked, new_dim)
    if name:
        arr = arr.rename(name)
    return arr, stack_info


def to_unstacked_dataset(arr: np.ndarray, stack_info: StackInfo) -> xr.Dataset:
    if type(arr) == xr.DataArray:
        arr = arr.values
    elif type(arr) == np.ndarray:
        pass
    else:
        raise RuntimeError('Passed array must be of type DataArray or ndarray')

    unstacked = {}
    curr_i = 0
    for var in stack_info:
        feature_len = 1
        unstacked_shape = [arr.shape[0], ]
        for dim_len in var.shape:
            feature_len *= dim_len
            unstacked_shape.append(dim_len)
        var_slice = arr[:, curr_i:curr_i+feature_len]
        var_unstacked = var_slice.reshape(unstacked_shape)
        unstacked[var.name] = xr.DataArray(var_unstacked, dims=var.dims)
        curr_i += feature_len
    ds = xr.Dataset(unstacked)
    return ds


def shuffle_dataset(ds: xr.Dataset, dim: str, seed=None) -> xr.Dataset:
    idx = np.arange(ds.dims[dim])
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(idx)
    return ds.isel({dim: idx})


def train_test_split_dataset(ds: xr.Dataset, dim: str,
                             train_size: Optional[Union[float, int]] = None,
                             test_size: Optional[Union[float, int]] = None,
                             shuffle=True, seed=None) -> tuple:
    if shuffle:
        ds = shuffle_dataset(ds, dim, seed=seed)
    count = ds.dims[dim]
    if train_size is None:
        assert test_size is not None
        if test_size > 1:
            test_count = int(test_size)
            assert test_count < count
        else:
            test_count = int(count * test_size)
            assert test_count >= 0
        train_count = count - test_count
    else:
        assert test_size is None
        if train_size > 1:
            train_count = int(train_size)
            assert train_count <= count
        else:
            train_count = int(count * train_size)
            assert train_count > 0
        test_count = count - train_count
    train = ds.isel({dim: slice(0, train_count)})
    test = ds.isel({dim: slice(train_count, None)})
    return train, test


def to_normalized_dataset(ds: xr.Dataset, stats_info: Optional[dict] = None) -> Union[xr.Dataset, dict]:
    """ Normalize quantities in a dataset by their mean and standard deviation.
    """
    stats_info = {}
    ds_normalized = xr.zeros_like(ds)
    for name in list(ds):
        stats_info[name] = {
            'mean': ds[name].mean(),
            'std': ds[name].std()
        }
        ds_normalized[name] = (ds[name] - stats_info[name]['mean']) \
            / stats_info[name]['std']
    return ds_normalized, stats_info


def to_unnormalized_dataset(ds: xr.Dataset, stats_info: dict) -> xr.Dataset:
    """ Recover a dataset of previously normalized quantities by their mean and standard deviation.
    """
    ds_unnormalized = xr.zeros_like(ds)
    for name in list(ds):
        ds_unnormalized[name] = ds[name] * stats_info[name]['std'] \
            + stats_info[name]['mean']
    return ds_unnormalized


def load_ds_inputs(proj_path, columns=slice(0, None)):
    """ Load and subset the input data used throughout the experiments
    """
    ds_inputs = xr.open_dataset(r'C:\Users\adity\OneDrive\Desktop\mscThesis\Code3\Thesis_FLow\marginalTailAdaptiveFlow\real_world_experiments\data\nwp_saf_profiles_in.nc')
    ds_inputs = compute_layer_cloud_optical_depth(ds_inputs)
    inputs_relevant = ['temperature_fl', 'pressure_hl']
    if USE_CLOUD_OPTICAL_DEPTH:
        inputs_relevant += ['layer_cloud_optical_depth']
    ds_inputs = ds_inputs[inputs_relevant].sel(column=columns)
    return ds_inputs

def compute_layer_cloud_optical_depth(ds: xr.Dataset) -> xr.Dataset:
    """ Compute per-layer profiles of cloud optical depth using SAF profile data.
    """
    # Constants
    g = 9.80665  # m s^{-2}
    rho_liquid = 1000  # kg m^{-3}
    rho_ice = 917  # kg m^{-3}
    d_pressure = ds['pressure_hl'].diff(
        'half_level').rename({'half_level': 'level'})

    optical_depth = (ds['q_liquid'] / (rho_liquid * ds['re_liquid']) +
                     ds['q_ice'] / (rho_ice * ds['re_ice'])) * d_pressure / g
    optical_depth = optical_depth.rename('layer_cloud_optical_depth')
    optical_depth.attrs = {
        'long_name': 'Layer cloud optical depth', 'units': '1'}
    return xr.merge([ds, optical_depth])

def plot_weather(data, args=""):
    '''
    Generates and saves plots of the temperature, pressure, and the cloud optical depth at different atmospheric levels.
    Compare with Figure 3 in the Paper.
    This script is based on the repository https://zenodo.org/record/4320795#.YpTFbXVBxhE  "Copula-based synthetic data augmentation for machine-learning emulators".

    The input has to be an xarray like so:
    data = xr.Dataset({"temperature_fl": xr.DataArray(data[:, :137], dims=["column", "level"]),
                       "pressure_hl": xr.DataArray(data[:, 137:194], dims=["column", "halflevel"]),
                       "layer_cloud_optical_depth": xr.DataArray(data[:, 194:], dims=["column", "level"])})
    '''
    d_names = {
        'temperature_fl': 'Dry-bulb air temperature in K',
        'pressure_hl': 'Atmospheric pressure in Pa',
        'layer_cloud_optical_depth': 'Cloud optical depth'
    }

    data['pressure_hl'] /= 100 # Convert Pa to hPa
    alpha = 1
    batch_size = None
    num_samples = min(200, len(data["pressure_hl"]))
    np.random.seed(100)
    id_samples = np.random.choice(len(data["pressure_hl"]), num_samples, replace=False)

    with_banddepth = True
    fig, ax = plt.subplots(3, 1, figsize=(8,16), squeeze=False, sharex=True)

    # Generate plots for each feature
    for idx, name in enumerate(d_names.keys()):
        y_lim = find_ylim([data[name][id_samples]])
        print(f'Plotting {name}')
        plot_lines(data[name][id_samples], d_names[name], ax.flat[idx], y_lim,
                   batch_size=batch_size, with_banddepth=with_banddepth, alpha=alpha)

    # Legend
    patches = []
    if with_banddepth:
        patches.append(
            Line2D([0], [0], color='#785EF0', lw=1, label=' 0 - 25 %'))
        patches.append(
            Line2D([0], [0], color='#A091E4', lw=1, label='25 - 50 %'))
        patches.append(
            Line2D([0], [0], color='#E7E2FB', lw=1, label='50 - 100 %'))
    else:
        patches.append(Line2D([0], [0], color='#785EF0', lw=1, label='$i$ᵗʰ profile'))
    plt.legend(handles=patches, loc='best')
    plt.xlabel("Atmospheric level")
    plt.tight_layout()

    # save locally as pdf
    Path("plots/").mkdir(parents=True, exist_ok=True)
    if args=="":
        plt.savefig("plots/weather.pdf")
    else:
        plt.savefig(f"plots/weather_flow{args.marginals}_lay{args.num_layers}_hid{args.num_hidden}_bl{args.num_blocks}.pdf") # todo how to specify this?

# plotting helpers:
def plot_lines(ds, y_label, ax, y_lim, hide_y_label=False, batch_size=None, with_banddepth=False, alpha=0.01):

    def compute_idx_range(idx_arr, min, max):
        idx_range = idx_arr[int(len(idx_arr) * min): int(len(idx_arr) * max)]
        return idx_range

    def compute_banddepth_idx(ds, ax):
        res = sm.graphics.fboxplot(ds, ax=ax)
        return res[2]

    if with_banddepth:
        depth_ixs = compute_banddepth_idx(ds, ax=ax)
        ax.clear()

        # error bars:
        yerr = ds.std("column")
        ymean = ds.mean("column")
        if ds.name!="pressure_hl":
            ax.fill_between(range(137), ymean - yerr, ymean + yerr, alpha=0.3)
        else:
            ax.fill_between(range(138), ymean - yerr, ymean + yerr, alpha=0.3)

        for i in compute_idx_range(depth_ixs, 0.50, 1):
            ds.sel(column=i).plot(c='#E7E2FB', ax=ax, alpha=alpha)
        for i in compute_idx_range(depth_ixs, 0.25, 0.50):
            ds.sel(column=i).plot(c='#A091E4', ax=ax, alpha=alpha)
        for i in compute_idx_range(depth_ixs, 0.0, 0.25):
            ds.sel(column=i).plot(c='#785EF0', ax=ax, alpha=alpha)
    else:
        ax.clear()
        # error bars:
        yerr = ds.std("column")
        ymean = ds.mean("column")
        try:
            ax.fill_between(range(137), ymean - yerr, ymean + yerr, alpha=0.3)
        except:
            ax.fill_between(range(138), ymean - yerr, ymean + yerr, alpha=0.3)
        for i in np.random.choice(ds.column, batch_size):
            ds.sel(column=i).plot(c='#785EF0', alpha=alpha, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    if hide_y_label:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    ax.set_ylim(y_lim)

def find_ylim(ds_list):
    bounds = []
    for ds in ds_list:
        bounds.append([ds.min(), ds.max()])
    return (np.min(bounds), np.max(bounds))
