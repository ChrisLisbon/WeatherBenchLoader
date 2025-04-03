import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch

grid_names = {5: '5.625',
              2: '2.8125',
              1: '1.40625'}


def form_grid(grid: int, donor_file:str):
    """
    Function to save coordinates grid as npy for different resolutions
    :param donor_file:
    :param grid: resolution - 5, 2, 1 are available
    """
    grid = grid_names[grid]
    dataset = nc.Dataset(donor_file)
    lats = np.array(dataset.variables['lat'])
    lons = np.array(dataset.variables['lon'])
    # coord[0] - lons, coord[1] - lats
    coord = np.meshgrid(lons, lats)
    plt.imshow(coord[0])
    plt.title('Longs')
    plt.colorbar()
    plt.show()
    plt.imshow(coord[1])
    plt.title('Lats')
    plt.colorbar()
    plt.show()
    np.save(f'grid_files/lats_{grid}.npy', coord[1])
    np.save(f'grid_files/lons_{grid}.npy', coord[0])


def form_weights_grid(grid:int):
    """
    Function to calculate weights matrix based on latitude (RMSE weighted metric)
    L - latitude weighting factor
    (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020MS002203)
    :param grid: resolution - 5, 2, 1 are available
    """
    grid = grid_names[grid]
    lat = np.load(f'grid_files/lats_{grid}.npy') * np.pi / 180
    L = np.cos(lat) / (np.sum(np.cos(lat)) / lat.size)
    plt.imshow(L)
    plt.colorbar()
    plt.xlabel('lons')
    plt.ylabel('lats')
    plt.title('L - latitude weighting factor')
    plt.tight_layout()
    plt.show()
    np.save(f'grid_files/weights_{grid}.npy', L)


def wrmse(predicted, target, weights):
    """
    Function for calculation weighted RMSE
    """
    return np.mean(((predicted - target) ** 2 * weights) ** 0.5)


def wmae(predicted, target, weights):
    """
    Function for calculation weighted MAE
    """
    return np.mean((abs(predicted - target) * weights) ** 0.5)


class WRMSE(torch.nn.Module):
    """
    Loss function for PyTorch model optimization
    Weighted on latitude MSE
    """
    def __init__(self, weights):
        super(WRMSE, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        sq = (output - target) ** 2
        w_sq = sq*self.weights
        loss = torch.mean(w_sq)
        # sqrt does not use for stable training
        return loss


class WACC(torch.nn.Module):
    def __init__(self, weights):
        """
        Adaptation original code to loss
        https://github.com/pangeo-data/WeatherBench/blob/c8f53a9b243453fef3edecf5fdc1150b6b0f8f32/src/score.py#L46
        :param weights: tensor with weights depend on latitude
        """
        super(WACC, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        clim = target.mean(axis=1) # mean by time axis
        clim = clim[:, None, :, :]
        anom_pred = output - clim
        anom_real = target - clim

        pred_peaks = anom_pred - anom_pred.mean(axis=1).mean(axis=0)
        real_peaks = anom_real - anom_real.mean(axis=1).mean(axis=0)

        acc = (
                (self.weights * pred_peaks * real_peaks).sum() /
                torch.sqrt(sum(self.weights * pred_peaks ** 2).sum() * (self.weights * real_peaks ** 2).sum()
                )
        )
        # use reversed metric for optimization
        return 1 - acc

