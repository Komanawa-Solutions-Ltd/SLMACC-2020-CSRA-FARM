"""
 Author: Matt Hanson
 Created: 27/01/2021 9:49 AM
 """
import numpy as np
import os
from scipy.stats import pearsonr, truncnorm
import netCDF4 as nc
from copy import deepcopy
import matplotlib.pyplot as plt
from warnings import warn

# todo test



class MovingBlockBootstrapGenerator(object):
    block_size = None
    mean, std, clip_min, clip_max, nblocksize = None, None, None, None, None
    block_array = None

    def __init__(self, input_data, blocktype, block, nsims, data_base_path, sim_len=None, nblocksize=None,
                 save_to_nc=True):
        """

        :param input_data: 1d array or dict of 1d arrays, the data to resample
        :param blocktype: one of: 'single': use a single block size, block must be an integer
                                  'list': use np.choice to sample from a list or 1d array block must be list like and 1d
                                  'truncnormal': use truncnormal to generate a sudo normal distribution of block sizes
                                                 block must be (mean, std, clip_min, clip_max)
        :param block: the block(s) to use, see blocktype above for input options
        :param nsims: the number of simulations for each month to create, nsims may be increased so that
                      nsims % nblocksize = 0
        :param data_base_path: directory to store the generator
        :param sim_len: size of sim to create:
                        None: create sims the same size as input data (for each data key),
                        int: create sims of length int
                        dict: dictionary of keys: length, where keys match input data
        :param nblocksize: the number of differnt block sizes to use required for blocktype: 'list' and 'truncnormal'
        :param save_to_nc: boolean if True save to nc and then open the dataset (don't keep all data in memory),
                           if False keep in memory as dataset.
        """
        self.input_data = {}
        self.sim_len = {}
        if isinstance(input_data, dict):
            self.keys = input_data.keys()
            self.key = None
            for k, v in input_data.items():

                t = np.atleast_1d(v)
                assert t.ndim == 1, 'input data values must be a 1d array'
                self.input_data[k] = deepcopy(t)
        else:
            self.key = 'one_value'
            self.keys = ['one_value']
            t = np.atleast_1d(input_data)
            assert t.ndim == 1, 'input data must be a 1d array'
            self.input_data[self.key] = deepcopy(t)

        for k in self.keys:
            if sim_len is None:
                self.sim_len[k] = len(t)
            elif isinstance(sim_len, int):
                self.sim_len[k] = sim_len
            elif isinstance(sim_len, dict):
                self.sim_len[k] = sim_len[k]
            else:
                raise ValueError('{} not an acceptable argument for sim len'.format(sim_len))

        self.save_to_nc = save_to_nc
        if not os.path.exists(data_base_path):
            os.makedirs(data_base_path)
        self.data_base_path = data_base_path

        assert isinstance(nsims, int)

        if blocktype == 'single':
            assert isinstance(block, int), 'block must be an integer if using a single block size'
            self.block_size = block
            self.nblocksize = 1
            self.data_id = '{}_{:02d}_{}'.format(blocktype, block, self.nsims)
            self.nsims = nsims


        elif blocktype == 'list':
            assert nblocksize is not None and isinstance(nblocksize, int)
            self.nblocksize = nblocksize
            block = np.atleast_1d(block).astype(int)
            assert block.ndim == 1, 'only 1d arrrays/options'
            self.block_array = block

            # match nsims and nblocksize
            t = int(round(nsims / nblocksize))
            self.nsims = nblocksize * t

            self.data_id = '{}_{}_{}_{}'.format(blocktype, '-'.join(block), nblocksize, self.nsims)

        elif blocktype == 'truncnormal':
            assert nblocksize is not None and isinstance(nblocksize, int)
            self.nblocksize = nblocksize
            assert len(block) == 4, 'with truncnormal block size must be 4 as it holds the 4 parameters'
            self.mean, self.std, self.clip_min, self.clip_max = block

            # match nsims and nblocksize
            t = int(round(nsims / nblocksize))
            self.nsims = nblocksize * t

            self.data_id = '{}_m{}_sd{}_c{}_c{}_s{}_{}'.format(blocktype, *block, nblocksize, nsims)

        else:
            raise ValueError('incorrect value for blocktype: {}'.format(blocktype))

        self.blocktype = blocktype
        self.datapath = os.path.join(self.data_base_path, '{}.nc'.format(self.data_id))

        if not os.path.exists(self.datapath):
            self._make_data()

        if save_to_nc:
            self.dataset = nc.Dataset(self.datapath)
        else:
            self.dataset = {}

    def _make_data(self):
        """
        make the dataset and save to a netcdf,
        :return:
        """
        # set blocks
        if self.blocktype == 'single':
            blocks = [self.block_size]
        elif self.blocktype == 'list':
            blocks = np.random.choice(self.block_array, self.nblocksize)
        elif self.blocktype == 'truncnormal':
            a, b = (self.clip_min - self.mean) / self.std, (self.clip_max - self.mean) / self.std
            blocks = truncnorm(a, b, loc=self.mean, scale=self.std).rvs(size=self.nblocksize).round().astype(int)
        else:
            raise ValueError('shouldnt get here')

        if self.save_to_nc:
            nd_data = nc.Dataset(self.datapath)
            nd_data.createDimension('sim_num', self.nsims)
            d = nd_data.createVariable('sim_num', int, ('sim_num',))
            d[:] = range(self.nsims)

        for k in self.keys:
            out = np.zeros((self.nsims, self.sim_len[k])) * np.nan
            l = self.nsims // self.nblocksize
            for i, b in enumerate(blocks):
                out[i * l:(i + 1) * l] = self._make_moving_sample(b, l)

            if self.save_to_nc:
                nd_data.createDimension('sim_len_{}'.format(k), self.sim_len[k])
                t = nd_data.createVariable(k, float, ('sim_len_{}'.format(k), 'sim_num'))
                t[:] = out.transpose()
                t = nd_data.createVariable('{}_mean'.format(k), float, ('sim_num',))
                t[:] = out.mean(axis=1)
            else:
                self.dataset[k] = out
                self.dataset['{}_mean'.format(k)] = out.mean(axis=1)

    def _make_moving_sample(self, block, nsamples):
        #todo
        raise NotImplementedError

    def plot_auto_correlation(self, nsims, lags, key=None):
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        # todo make auto correlation comparison with some number of samples.
        raise NotImplementedError

    def plot_means(self, key=None, bins='freedman', show=True):
        """
        plot up a histogram of the means, 10000 samples
        :param key: key to plot means of (if None set to self.key)
        :param bins: 'freedman', or int, number of bins to use, by default use Freedman–Diaconis rule
        :param show: boolean, if True call plt.show()
        :return:
        """
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        assert key in self.keys
        data = self.get_data('{}_mean'.format(key), 10000, 'any', warn_level=1)
        fig, ax = plt.subplots()
        if bins =='freedman':
            bins = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / 10000 ** 1 / 3
        ax.hist(data, bins=bins)
        if show:
            plt.show()
        else:
            return fig, ax

    def get_data(self,  nsims, key=None, mean='any', tolerance=None, warn_level=0.1):
        """
        pull simulations from the dataset, samples bootstrap simulations with replacement.
        :param key: data key to pull from, (if None set to self.key),
                    self.key will be None if there is more than one key
        :param nsims: the number of simulations to pull
        :param mean: one of 'any': select nsims from full bootstrap
                            float: select nsims only from data which satisfies
                                   np.isclose(simulation_means, mean, atol=tolerance, rtol=0)
        :param tolerance: None or float, seem mean for use
        :param warn_level: where warn_level of the nsamples must be replacements,
        :return:
        """
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        assert key in self.keys
        if mean == 'any':
            idxs = np.random.choice(range(self.nsims), (nsims,))
            if self.save_to_nc:
                out = np.array(self.dataset.variables[key][:, idxs])
                return out
            else:
                out = self.dataset[idxs]
                return out

        else:
            if self.save_to_nc:
                means = np.array(self.dataset.variables['{}_mean'.format(key)])
                idxs = np.where(np.isclose(means, mean, atol=tolerance, rtol=0))[0]
                if len(idxs) < (1 - warn_level) * nsims:
                    warn('selecting {} from {} unique simulations, less than '
                         'warn_level: {} of repetition'.format(nsims, len(idxs), warn_level))
                idxs = np.random.choice(idxs, nsims)
                out = np.array(self.dataset.variables[key][:, idxs])
                return out
            else:
                means = self.dataset['{}_mean'.format(key)]
                idxs = np.where(np.isclose(means, mean, atol=tolerance, rtol=0))[0]
                if len(idxs) <= (1 - warn_level) * nsims:
                    warn('selecting {} from {} unique simulations, less than '
                         'warn_level: {} of repetition'.format(nsims, len(idxs), warn_level))
                idxs = np.random.choice(idxs, nsims)
                out = self.dataset[idxs]
                return out


def calc_autocorrelation(x, lags=30):  # todo may need to re group to speed up data
    data = []
    size = len(x)
    for l in range(lags):
        r, p = pearsonr(x[0:size - l], x[l:]),  # todo perhaps use np version
        data.append(r)
    return data
