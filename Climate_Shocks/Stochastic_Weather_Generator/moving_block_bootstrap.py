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
from matplotlib.cm import get_cmap
from warnings import warn
import datetime


class MovingBlockBootstrapGenerator(object):
    block_size = None
    mean, std, clip_min, clip_max, nblocksize = None, None, None, None, None
    block_array = None
    datapath = None

    def __init__(self, input_data, blocktype, block, nsims, data_path=None, sim_len=None, nblocksize=None,
                 save_to_nc=True, comments=''):
        """

        :param input_data: 1d array or dict of 1d arrays, the data to resample
        :param blocktype: one of: 'single': use a single block size, block must be an integer
                                  'list': use np.choice to sample from a list or 1d array block must be list like and 1d
                                  'truncnormal': use truncnormal to generate a sudo normal distribution of block sizes
                                                 block must be (mean, std, clip_min, clip_max) note that blocks are not
                                                 mixed and matched (e.g. each sim will be created from only one
                                                 block size
        :param block: the block(s) to use, see blocktype above for input options as either this or as a dictionary
                      with same keys as input data
        :param nsims: the number of simulations for each month to create, nsims may be increased so that
                      nsims % nblocksize = 0
        :param data_path: path to store the generator as a netcdf file (.nc)
        :param sim_len: size of sim to create:
                        None: create sims the same size as input data (for each data key),
                        int: create sims of length int
                        dict: dictionary of keys: length, where keys match input data
        :param nblocksize: the number of differnt block sizes to use required for blocktype: 'list' and 'truncnormal'
                           one value or dictionary
        :param save_to_nc: boolean if True save to nc and then open the dataset (don't keep all data in memory),
                           if False keep in memory as dataset.
        :param comments: str other comments to save to the netcdf file
        """
        self.comments = comments
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
                self.sim_len[k] = len(self.input_data[k])
            elif isinstance(sim_len, int):
                self.sim_len[k] = sim_len
            elif isinstance(sim_len, dict):
                self.sim_len[k] = sim_len[k]
            else:
                raise ValueError('{} not an acceptable argument for sim len'.format(sim_len))
            temp_len = len(self.input_data[k])
            temp_sim_len = self.sim_len[k]
            assert (temp_len % temp_sim_len) == 0, ('input data length: {} is not a multiple of simlenth {} '
                                                    'for key {}'.format(temp_len, temp_sim_len, k))
        self.save_to_nc = save_to_nc
        if save_to_nc:
            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            self.datapath = data_path

        assert isinstance(nsims, int)

        temp = deepcopy(block)
        if not isinstance(block, dict):
            block = {}
            for k in self.keys:
                block[k] = temp

        temp = deepcopy(nblocksize)
        if not isinstance(temp, dict):
            nblocksize = {}
            for k in self.keys:
                nblocksize[k] = temp

        if blocktype == 'single':
            for k in self.keys:
                assert isinstance(block[k], int), 'block must be an integer if using a single block size'
            self.block_size = block
            self.nblocksize = {k: 1 for k in self.keys}
            temp = 'block type: {}\n'.format(blocktype, block, self.nsims)
            self.nsims = {k: nsims for k in self.keys}

            self.data_id = temp + '\n'.join(['key: {} number of sims: {} block: {:04d}'.format(e, self.nsims[e],
                                                                                               block[e]) for e in
                                             self.keys])


        elif blocktype == 'list':

            for k in self.keys:
                assert nblocksize[k] is not None and isinstance(nblocksize[k], int)
                block[k] = np.atleast_1d(block[k]).astype(int)
                assert block[k].ndim == 1, 'only 1d arrrays/options'
            self.nblocksize = nblocksize
            self.block_array = block

            # match nsims and nblocksize
            self.nsims = {nblocksize[k] * int(round(nsims / nblocksize[k])) for k in self.keys}

            self.data_id = ('block type: {}'.format(blocktype) +
                            '\n'.join([
                                'key:{} blocks: {} nblocksize: {} number of sims:{}'.format(e, block[e],
                                                                                            self.nblocksize[e],
                                                                                            self.nsims[e]) for e in
                                self.keys]
                            ))

        elif blocktype == 'truncnormal':
            self.mean, self.std, self.clip_min, self.clip_max = {}, {}, {}, {}
            for k in self.keys:
                assert nblocksize[k] is not None and isinstance(nblocksize[k], int)
                assert len(block[k]) == 4, 'with truncnormal block size must be 4 as it holds the 4 parameters'
                self.mean[k], self.std[k], self.clip_min[k], self.clip_max[k] = block[k]
            self.nblocksize = nblocksize

            # match nsims and nblocksize
            self.nsims = {k: nblocksize[k] * int(round(nsims / nblocksize[k])) for k in self.keys}

            temp = [('key: {} mean: {} '
                     'stdev: {} min cutoff: {} '
                     'max cutoff: {} '
                     'number of block sizes: {} '
                     'number of sims: {}').format(e, *block[e], self.nblocksize[e], self.nsims[e]) for e in
                    self.keys]

            self.data_id = 'block type:{}\n' + '\n'.join(temp)
        else:
            raise ValueError('incorrect value for blocktype: {}'.format(blocktype))

        self.blocktype = blocktype

        if save_to_nc:
            if not os.path.exists(self.datapath):
                self._make_data()
            self.dataset = nc.Dataset(self.datapath)
        else:
            self.dataset = {}
            self._make_data()

    def _make_data(self):
        """
        make the dataset and save to a netcdf,
        :return:
        """
        # set blocks
        blocks = {}
        for k in self.keys:
            if self.blocktype == 'single':
                blocks = [self.block_size[k]]
            elif self.blocktype == 'list':
                blocks = np.random.choice(self.block_array[k], self.nblocksize[k])
            elif self.blocktype == 'truncnormal':
                a, b = (self.clip_min[k] - self.mean[k]) / self.std[k], (self.clip_max[k] - self.mean[k]) / self.std[k]
                blocks = truncnorm(a, b, loc=self.mean[k],
                                   scale=self.std[k]).rvs(size=self.nblocksize[k]).round().astype(int)
            else:
                raise ValueError('shouldnt get here')

        if self.save_to_nc:
            nd_data = nc.Dataset(self.datapath, 'w')
            nd_data.description = self.data_id
            nd_data.created = datetime.datetime.now().isoformat()
            nd_data.comments = self.comments
            nd_data.script = __file__
            for k in self.keys:
                nd_data.createDimension('sim_num_{}'.format(k), self.nsims[k])
                d = nd_data.createVariable('sim_num_{}'.format(k), int, ('sim_num_{}'.format(k),))
                d[:] = range(self.nsims[k])

        for k in self.keys:
            print(f'making data for {k}')
            out = np.zeros((self.nsims[k], self.sim_len[k])) * np.nan
            n = self.nsims[k] // self.nblocksize[k]

            for i, b in enumerate(blocks):
                out[i * n:(i + 1) * n] = self._make_moving_sample(k, b, n)

            if self.save_to_nc:
                nd_data.createDimension('sim_len_{}'.format(k), self.sim_len[k])
                t = nd_data.createVariable(k, float, ('sim_len_{}'.format(k), 'sim_num_{}'.format(k)))
                t[:] = out.transpose()  # this is backwards to manage quick reading
                t = nd_data.createVariable('{}_mean'.format(k), float, ('sim_num_{}'.format(k),))
                t[:] = out.mean(axis=1)
            else:
                self.dataset[k] = out
                self.dataset['{}_mean'.format(k)] = out.mean(axis=1)

    def _make_moving_sample(self, key, blocksize, nsamples):
        usable_len = len(self.input_data[key]) - blocksize
        possible_idxs = range(usable_len)

        num_per_sample = -(-self.sim_len[key] // blocksize)
        start_idxs = np.random.choice(possible_idxs, nsamples * num_per_sample)
        num_select = num_per_sample * blocksize
        idxs = np.array([np.arange(e, e + blocksize) for e in start_idxs]).flatten()
        out = self.input_data[key][idxs].reshape((nsamples, num_select))
        out = out[:, 0:self.sim_len[key]]
        return out

    def plot_auto_correlation(self, nsims, lags, key=None, quantiles=(5, 25), alpha=0.5, show=True, hlines=[0, 0.5]):
        """

        :param nsims: number of new simulations to select
        :param lags: number of steps of autocorrelation to calculate
        :param key: key to data or None, note None will only select datasets is there is only one key
        :param quantiles: symetrical quantiles (0-50) to plot confidence interval on.
        :param alpha: the alpha to plot the quantiles
        :param show: bool if True call plt.show() otherwise return fig, ax
        :param hlines: list, plot hlines at each value
        :return:
        """
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        sim_data = self.get_data(nsims=nsims, key=key, max_replacement_level=1)
        org_data = self.get_org_data(key=key)
        org_plot = np.zeros((org_data.shape[0], lags)) * np.nan
        sim_plot = np.zeros((nsims, lags)) * np.nan
        for i in range(nsims):
            sim_plot[i] = calc_autocorrelation(sim_data[i], lags)
        for j in range(org_plot.shape[0]):
            org_plot[j] = calc_autocorrelation(org_data[j], lags)
        fig, ax = plt.subplots()

        # plot the quantiles
        x = range(lags)
        for data, cmap, c, label in zip([org_plot, sim_plot], [get_cmap('Reds'), get_cmap('Blues')],
                                        ['r', 'b'], ['input', 'sim']):
            for i, q in enumerate(quantiles):
                cuse = cmap((i + 1) / (len(quantiles) + 2))
                ax.fill_between(x, np.nanpercentile(data, q, axis=0), np.nanpercentile(data, 100 - q, axis=0),
                                alpha=alpha, color=cuse, label='{}-quant:{}-{}'.format(label, q, 100 - q))

            # plot the medians
            ax.plot(x, np.nanmedian(data, axis=0), color=c, label='{}-med'.format(label))

        for l in hlines:
            ax.axhline(l, linestyle='--', color='k', alpha=0.5)
        ax.set_xlabel('autocorrelation lag'.format(key))
        ax.set_ylabel('pearson r')
        ax.legend()
        ax.set_title(key)
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_means(self, key=None, bins=100, show=True, include_input=True, density=True):
        """
        plot up a histogram of the means,
        :param key: key to plot means of (if None set to self.key)
        :param bins: 'int, number of bins to use
        :param show: boolean, if True call plt.show()
        :param include_input: boolean if True add the original data and set alpha to 0.5 for both
        :param density: boolean see density kwarg for matplotlib.pyplot.hist()
        :return:
        """
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        assert key in self.keys
        data = self.get_means(key=key)
        fig, ax = plt.subplots()
        ax.hist(data, bins=bins, label='resampled data', alpha=1 - 0.5 * include_input, density=density, color='b')
        if include_input:
            org_data = self.get_org_data(key).mean(axis=1)
            ax.hist(org_data, bins, label='original data', alpha=0.5, density=density, color='r')
        ax.set_xlabel('mean values for: {}'.format(key))
        if density:
            ax.set_ylabel('fraction')
        else:
            ax.set_ylabel('count')
        ax.legend()
        ax.set_title(key)
        if show:
            plt.show()
        else:
            return fig, ax

    def get_org_data(self, key=None):
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        output = deepcopy(self.input_data[key])
        shape = (len(output) // self.sim_len[key], self.sim_len[key])
        output = output.reshape(shape)
        return output

    def get_means(self, key=None):
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        assert key.replace('_mean', '') in self.keys, 'key: {} not found'.format(key)

        if self.save_to_nc:
            out = np.array(self.dataset.variables['{}_mean'.format(key)][:]).transpose()
            return out
        else:
            out = self.dataset['{}_mean'.format(key)]
            return out

    def get_data(self, nsims, key=None, mean='any', tolerance=None, lowerbound=None, upper_bound=None,
                 max_replacement_level=0.1,
                 under_level='warn'):
        """
        pull simulations from the dataset, samples bootstrap simulations with replacement.
        :param key: data key to pull from, (if None set to self.key),
                    self.key will be None if there is more than one key
        :param nsims: the number of simulations to pull, pulls using np.choice, so may resample
        :param mean: one of:
                            'any': select nsims from full bootstrap
                            None: select nsims using upper and lower bounds
                            float: select nsims only from data which satisfies
                                   np.isclose(simulation_means, mean, atol=tolerance, rtol=0)
        :param tolerance: None or float, seem mean for use
        :param upper_bound: float, select data whose mean is <= this bound
        :param lowerbound: float, select data whose mean is >= this bound
        :param max_replacement_level: warn,raise, or pass when at least
                                      max_replacement_level * nsims must be replacements,
        :param under_level: describes the action to be taken 'warn', 'raise', 'pass'
        :return: data shape (samples, simlen)
        """
        # manage key
        if key is None:
            if self.key is None:
                raise ValueError('more than one key in the dataset, please provide key')
            key = deepcopy(self.key)
        assert key in self.keys, 'key: {} not found'.format(key)

        # define the indexes to pull
        if mean == 'any':
            idxs = np.random.choice(range(self.nsims[key]), (nsims,))
            if len(idxs) == 0:
                raise ValueError('no simulations for key: {}'.format(key))
        else:
            means = self.get_means(key)
            if mean is None:
                if upper_bound is None or lowerbound is None:
                    raise ValueError('if mean is None then upper and lower bound must not be None')
                idxs = np.where((means >= lowerbound) & (means <= upper_bound))[0]
                if len(idxs) == 0:
                    raise ValueError(
                        'no simulations between {}<=x<={} for key: {}'.format(lowerbound, upper_bound, key))
            else:
                if tolerance is None:
                    raise ValueError('tolerance must not be None if pulling with means (mean is a float)')
                idxs = np.where(np.isclose(means, mean, atol=tolerance, rtol=0))[0]
                if len(idxs) == 0:
                    raise ValueError(
                        'no simulations matching mean: {} and tolerance:{} for key: {}'.format(mean, tolerance, key))

            if len(idxs) <= (1 - max_replacement_level) * nsims:
                mess = ('{}: selecting {} from {} unique simulations, less than '
                        'warn_level: {} of repetition'.format(key, nsims, len(idxs), max_replacement_level))
                if under_level == 'warn':
                    warn(mess)
                elif under_level == 'raise':
                    raise ValueError(mess)
                elif under_level == 'pass':
                    pass
                else:
                    raise ValueError("incorrect arg for under_level expected on of ['warn', 'raise', 'pass']")
            idxs = np.random.choice(idxs, nsims)

        # pull data
        if self.save_to_nc:
            out = np.array(self.dataset.variables[key][:, idxs]).transpose()
            temp = out.mean()
        else:
            out = self.dataset[idxs]
            temp = out.mean()

        # a check if pulled out with means
        if mean != 'any' and (temp > (mean + tolerance) or temp < (mean - tolerance)):
            warn('sampled mean {} is outside of the set mean: {} and tolerance {}'.format(temp, mean, tolerance))

        return out


def calc_autocorrelation(x, lags=30):
    data = np.zeros((lags,))
    size = len(x)
    for l in range(lags):
        r, p = pearsonr(x[0:size - l], x[l:])
        data[l] = r
    return data


if __name__ == '__main__':
    pass  # todo write stand alone test!
