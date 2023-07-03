"""
created matt_dumont 
on: 26/06/23
"""
import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import netCDF4 as nc
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class BaseSimpleFarmModel(object):
    # create dictionary of attribute name to attribute long name (unit)
    attr_dict = {
        'state': 'farm state (none)',
        'feed': 'feed on farm (kgDM/ha)',
        'money': 'profit/loss (NZD)',
        'feed_demand': 'feed demand (kgDM/ha)',
        'prod': 'product produced (kg/ha)',
        'prod_money': 'profit from product (NZD)',
        'feed_imported': 'supplemental feed imported (kgDM/ha)',
        'feed_cost': 'cost of feed imported (NZD)',
        'running_cost': 'general farm running cost (ex. feed beyond expected feed) (NZD)',
        'debt_servicing': 'debt servicing (NZD)',
        'pg': 'pasture growth (kgDM/ha)',
        'product_price': 'product price (NZD/kg)',
        'feed_price': 'cost of feed imported (NZD)'

    }
    # create dictionary of attribute name to object name
    obj_dict = {
        'state': 'model_state',
        'feed': 'model_feed',
        'money': 'model_money',
        'feed_demand': 'model_feed_demand',
        'prod': 'model_prod',
        'prod_money': 'model_prod_money',
        'feed_imported': 'model_feed_imported',
        'feed_cost': 'model_feed_cost',
        'running_cost': 'model_running_cost',
        'debt_servicing': 'model_debt_service',
        'pg': 'pg',
        'product_price': 'product_price',
        'feed_price': 'sup_feed_cost',
    }
    inpath = None

    states = None  # dummy value, must be set in child class
    month_reset = None  # dummy value, must be set in child class

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price, monthly_input=True):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number or np.ndarray shape (nsims,)
        :param imoney: initial money number or np.ndarray shape (nsims,)
        :param sup_feed_cost: cost of supplementary feed $/kgDM or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param prod_price: income price $/kg product or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param monthly_input: if True, monthly input, if False, daily input
        """
        # define model shape
        assert set(all_months).issubset(set(range(1, 13))), f'months must be in range 1-12'
        self.month_len = len(all_months)
        all_month_org = deepcopy(all_months)
        self.time_len = np.sum([month_len[m] for m in all_months])
        self.nsims = len(istate)
        self.model_shape = (self.time_len + 1, self.nsims)
        self._run = False

        assert all(len(e) == self.nsims for e in (ifeed, imoney))
        if monthly_input:
            all_months = np.concatenate([np.repeat(m, month_len[m]) for m in all_month_org])
        if monthly_input:
            all_days = np.concatenate([np.arange(1, month_len[m] + 1) for m in all_month_org])
        else:
            all_days = pd.Series(all_months == np.concatenate(([all_months[0]], all_months[:-1])))
            all_days[0] = False
            all_days = all_days.cumsum() - (all_days.cumsum().where(~all_days).ffill().fillna(0).astype(int))
            all_days = all_days.values + 1
        all_year = np.cumsum(all_months == 1)
        ifeed = np.atleast_1d(ifeed)
        imoney = np.atleast_1d(imoney)

        # handle pg input options
        pg = np.atleast_1d(pg)
        if pg.ndim == 1:
            assert len(pg) == self.month_len, f'pg must be length time: {self.month_len=}, got: {len(pg)=}'
            if monthly_input:
                pg = np.concatenate([np.repeat(p, month_len[m]) for p, m in zip(pg, all_month_org)])
            pg = np.repeat(pg[:, np.newaxis], self.nsims, axis=1)
        elif monthly_input:
            pg = np.concatenate([np.repeat(p[np.newaxis], month_len[m], axis=0) for p, m in zip(pg, all_month_org)])
        self.pg = np.concatenate([pg[[0]], pg])
        assert self.pg.shape == self.model_shape, f'pg shape must be: {self.model_shape=}, got: {pg.shape=}'

        # handle sup_feed_cost input options
        if pd.api.types.is_number(sup_feed_cost):
            sup_feed_cost = np.full(self.model_shape, sup_feed_cost)
        elif sup_feed_cost.ndim == 1:
            assert len(sup_feed_cost) == self.month_len, (f'sup_feed_cost must be length time: {self.month_len=}, '
                                                          f'got: {len(sup_feed_cost)=}')
            if monthly_input:
                sup_feed_cost = np.concatenate(
                    [np.repeat(e, month_len[m]) for e, m in zip(sup_feed_cost, all_month_org)])
            sup_feed_cost = np.repeat(sup_feed_cost[:, np.newaxis], self.nsims, axis=1)
        elif monthly_input:
            sup_feed_cost = np.concatenate(
                [np.repeat(e[np.newaxis], month_len[m], axis=0) for e, m in zip(sup_feed_cost, all_month_org)])
        self.sup_feed_cost = np.concatenate([sup_feed_cost[[0]], sup_feed_cost], axis=0)
        assert self.sup_feed_cost.shape == self.model_shape, (f'sup_feed_cost shape must be: {self.model_shape=}, '
                                                              f'got: {sup_feed_cost.shape=}')

        # manage product price options
        if pd.api.types.is_number(product_price):
            product_price = np.full(self.model_shape, product_price)
        elif product_price.ndim == 1:
            assert len(product_price) == self.month_len, (f'product_price must be length time: {self.time_len=}, '
                                                          f'got: {len(product_price)=}')
            if monthly_input:
                product_price = np.concatenate(
                    [np.repeat(e, month_len[m]) for e, m in zip(product_price, all_month_org)])
            product_price = np.repeat(product_price[:, np.newaxis], self.nsims, axis=1)
        elif monthly_input:
            product_price = np.concatenate(
                [np.repeat(e[np.newaxis], month_len[m], axis=0) for e, m in zip(product_price, all_month_org)])

        self.product_price = np.concatenate([product_price[[0]], product_price], axis=0)
        assert self.product_price.shape == self.model_shape, (f'product_price shape must be: {self.model_shape=}, '
                                                              f'got: {product_price.shape=}')

        assert set(istate).issubset(
            set(self.states.keys())), f'unknown istate: {set(istate)} must be one of {self.states}'

        # setup key model values
        self.model_state = np.full(self.model_shape, -1, dtype=int)
        self.model_feed = np.full(self.model_shape, np.nan)
        self.model_money = np.full(self.model_shape, np.nan)

        # setup output values
        self.model_feed_demand = np.full(self.model_shape, np.nan)
        self.model_prod = np.full(self.model_shape, np.nan)
        self.model_prod_money = np.full(self.model_shape, np.nan)
        self.model_feed_imported = np.full(self.model_shape, np.nan)
        self.model_feed_cost = np.full(self.model_shape, np.nan)
        self.model_running_cost = np.full(self.model_shape, np.nan)
        self.model_debt_service = np.full(self.model_shape, np.nan)

        # set initial values
        self.model_state[0, :] = istate
        self.model_feed[0, :] = ifeed
        self.model_money[0, :] = imoney
        self.all_months = np.concatenate([[0], all_months])
        self.all_days = np.concatenate([[0], all_days])
        self.all_year = np.concatenate([[0], all_year])

        # internal check of shapes
        for v in self.obj_dict.values():
            assert v.shape == self.model_shape, f'bad shape for {v} {v.shape=} must be {self.model_shape=}'
        # ntime shapes
        for v in [self.all_months, self.all_days, self.all_year]:
            assert v.shape == (
                self.model_shape[0],), f'bad shape for {v} {v.shape=} must be {(self.model_shape[0],)}'

    def run_model(self):
        """
        run the model
        :return:
        """
        assert not self._run, f'model has already been run'
        for i_month in range(1, self.time_len + 1):
            month = self.all_months[i_month]
            day = self.all_days[i_month]

            # set start of day values
            current_money = deepcopy(self.model_money[i_month - 1, :])
            current_feed = deepcopy(self.model_feed[i_month - 1, :])
            current_state = deepcopy(self.model_state[i_month - 1, :])

            # pasture growth
            current_feed += self.pg[i_month, :]

            # feed cattle
            feed_needed = self.calculate_feed_needed(month, current_state)
            current_feed = current_feed - feed_needed
            self.model_feed_demand[i_month, :] = feed_needed

            # produce product
            produced_product = self.calculate_production(month, current_state)
            self.model_prod[i_month, :] = produced_product

            # sell product
            prod_money = produced_product * self.product_price[i_month, :]
            self.model_prod_money[i_month, :] = prod_money
            current_money += prod_money

            next_state = self.calculate_next_state(month, current_state)

            sup_feed = self.calculate_sup_feed(month, current_state)
            self.model_feed_imported[i_month, :] = sup_feed
            current_feed += sup_feed

            sup_feed_cost = sup_feed * self.sup_feed_cost[i_month, :]
            self.model_feed_cost[i_month, :] = sup_feed_cost
            current_money -= sup_feed_cost

            # add running_cost
            run_costs = self.calculate_running_cost(month, current_state)
            self.model_running_cost[i_month, :] = run_costs
            current_money -= run_costs

            # add debt servicing
            debt_servicing = self.calculate_debt_servicing(month, current_state)
            self.model_debt_service[i_month, :] = debt_servicing
            current_money -= debt_servicing

            # new year? reset state
            if month == self.month_reset and day == 1:
                next_state = self.reset_state()

            # set key values
            self.model_state[i_month, :] = next_state
            self.model_feed[i_month, :] = current_feed
            self.model_money[i_month, :] = current_money

        self._run = True

    def save_results_nc(self, outpath):
        """
        save results to netcdf file
        :param outpath: path to save file
        :return:
        """
        assert self._run, 'model must be run before saving results'
        assert isinstance(outpath, Path) or isinstance(outpath,
                                                       str), f'outpath must be Path or str, got: {type(outpath)}'
        outpath = Path(outpath)
        outpath.parent.mkdir(exist_ok=True, parents=True)

        with nc.Dataset(outpath, 'w') as ds:
            ds.createDimension('time', self.time_len + 1)
            ds.createDimension('nsims', self.nsims)

            ds.createVariable('sim', 'i4', ('nsims',))
            ds.variables['sims'][:] = range(self.nsims)
            ds.variables['sims'].setncattr('long_name', 'simulation number')

            ds.createVariable('month', 'i4', ('time',))
            ds.variables['month'][:] = self.all_months
            ds.variables['month'].setncattr('long_name', 'month of the year')

            ds.createVariable('year', 'i4', ('nsims',))
            ds.variables['year'].setncattr('long_name', 'year')
            ds.variables['year'][:] = self.all_year

            ds.createVariable('day', 'i4', ('nsims',))
            ds.variables['day'].setncattr('long_name', 'day of the month')
            ds.variables['day'][:] = self.all_days

            ds.createVariable('state', 'i4', ('time', 'nsims'))
            ds.variables['state'].setncattrs({'long_name': 'farm state', 'units': 'none'})
            ds.variables['state'][:] = self.model_state

            ds.createVariable('feed', 'f8', ('time', 'nsims'))
            ds.variables['feed'].setncattrs({'long_name': 'feed on farm', 'units': 'kgDM/ha'})
            ds.variables['feed'][:] = self.model_feed

            ds.createVariable('money', 'f8', ('time', 'nsims'))
            ds.variables['money'].setncattrs({'long_name': 'profit/loss', 'units': 'NZD'})
            ds.variables['money'][:] = self.model_money

            ds.createVariable('feed_demand', 'f8', ('time', 'nsims'))
            ds.variables['feed_demand'].setncattrs({'long_name': 'feed demand', 'units': 'kgDM/ha'})
            ds.variables['feed_demand'][:] = self.model_feed_demand

            ds.createVariable('prod', 'f8', ('time', 'nsims'))
            ds.variables['prod'].setncattrs({'long_name': 'product produced', 'units': 'kg/ha'})
            ds.variables['prod'][:] = self.model_prod

            ds.createVariable('prod_money', 'f8', ('time', 'nsims'))
            ds.variables['prod_money'].setncattrs({'long_name': 'profit from product', 'units': 'NZD'})
            ds.variables['prod_money'][:] = self.model_prod_money

            ds.createVariable('feed_imported', 'f8', ('time', 'nsims'))
            ds.variables['feed_imported'].setncattrs({'long_name': 'supplemental feed imported', 'units': 'kgDM/ha'})
            ds.variables['feed_imported'][:] = self.model_feed_imported

            ds.createVariable('feed_cost', 'f8', ('time', 'nsims'))
            ds.variables['feed_cost'].setncattrs({'long_name': 'cost of feed imported', 'units': 'NZD'})
            ds.variables['feed_cost'][:] = self.model_feed_cost

            ds.createVariable('running_cost', 'f8', ('time', 'nsims'))
            ds.variables['running_cost'].setncattrs(
                {'long_name': 'general farm running cost (ex. feed beyond expected feed)', 'units': 'NZD'})
            ds.variables['running_cost'][:] = self.model_running_cost

            ds.createVariable('debt_servicing', 'f8', ('time', 'nsims'))
            ds.variables['debt_servicing'].setncattrs(
                {'long_name': 'debt servicing', 'units': 'NZD'})
            ds.variables['debt_servicing'][:] = self.model_debt_service

            ds.createVariable('pg', 'f8', ('time', 'nsims'))
            ds.variables['pg'].setncattrs({'long_name': 'pasture growth', 'units': 'kgDM/ha'})
            ds.variables['pg'][:] = self.pg

            ds.createVariable('product_price', 'f8', ('time', 'nsims'))
            ds.variables['product_price'].setncattrs({'long_name': 'product price', 'units': 'NZD/kg'})
            ds.variables['product_price'][:] = self.product_price

            ds.createVariable('feed_price', 'f8', ('time', 'nsims'))
            ds.variables['feed_price'].setncattrs({'long_name': 'feed price', 'units': 'NZD/kg'})
            ds.variables['feed_price'][:] = self.sup_feed_cost

    @classmethod
    def from_input_file(self, inpath):
        """
        read in a model from a netcdf file
        :param inpath:
        :return:
        """
        assert isinstance(inpath, Path) or isinstance(inpath,
                                                      str), f'inpath must be Path or str, got: {type(inpath)}'
        inpath = Path(inpath)
        if not inpath.exists():
            raise FileNotFoundError(f'file does not exist: {inpath}')

        with nc.Dataset(inpath, 'r') as ds:
            all_months = np.array(ds.variables['month'][:])
            pg = np.array(ds.variables['pg'][:])
            product_price = np.array(ds.variables['product_price'][:])
            model_state = np.array(ds.variables['state'][:])
            model_feed = np.array(ds.variables['feed'][:])
            model_money = np.array(ds.variables['money'][:])
            sup_feed_cost = np.array(ds.variables['feed_price'][:])

            out = self(all_months=all_months, istate=model_state[0], pg=pg[1:],
                       ifeed=model_feed[0], imoney=model_money[0], sup_feed_cost=sup_feed_cost[1:],
                       product_price=product_price[1:], monthly_input=False)
            out.model_state = model_state
            out.model_feed = model_feed
            out.model_money = model_money

            out.model_feed_demand = np.array(ds.variables['feed_demand'][:])
            out.model_prod = np.array(ds.variables['prod'][:])
            out.model_prod_money = np.array(ds.variables['prod_money'][:])
            out.model_feed_imported = np.array(ds.variables['feed_imported'][:])
            out.model_feed_cost = np.array(ds.variables['feed_cost'][:])
            out.model_running_cost = np.array(ds.variables['running_cost'][:])
            out.model_debt_service = np.array(ds.variables['debt_servicing'][:])

        # set run status
        out._run = True
        out.inpath = inpath
        return out

    def save_results_csv(self, outdir, sims=None):
        """
        save results to csv in outdir with name sim_{sim}.csv
        :param outdir: output directory
        :param sims: sim numbers to save, if None, save all
        :return:
        """
        assert self._run, 'model must be run before saving results'
        assert isinstance(outdir, Path) or isinstance(outdir, str), f'outdir must be Path or str, got: {type(outdir)}'
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        if sims is None:
            sims = range(self.nsims)
        else:
            sims = np.atleast_1d(sims)

        for sim in sims:
            outdata = pd.DataFrame({
                'year': np.cumsum(self.all_months == self.month_reset) + 1,
                'month': self.all_months,
                'state': self.model_state[:, sim],
                'feed': self.model_feed[:, sim],
                'money': self.model_money[:, sim],
                'feed_demand': self.model_feed_demand[:, sim],
                'prod': self.model_prod[:, sim],
                'prod_money': self.model_prod_money[:, sim],
                'feed_imported': self.model_feed_imported[:, sim],
                'feed_cost': self.model_feed_cost[:, sim],
                'running_cost': self.model_running_cost[:, sim],
                'debt_service': self.model_debt_service[:, sim],
                'pg': self.pg[:, sim],
                'product_price': self.product_price[:, sim],
                'sup_feed_cost': self.sup_feed_cost[:, sim],
            })
            outdata.to_csv(outdir.joinpath(f'sim_{sim}.csv'), index=False)

    def plot_results(self, *ys, x='time', sims=None, mult_as_lines=True, twin_axs=False, figsize=(10, 8),
                     start_year=2000):
        """
        plot results
        :param ys: one or more of:
            - 'state' (state of the system)
            - 'feed' (feed on farm)
            - 'money' (money on farm)
            - 'feed_demand' (feed demand)
            - 'prod' (production)
            - 'prod_money' (production money)
            - 'feed_imported' (feed imported)
            - 'feed_cost' (feed cost)
            - 'running_cost' (running cost)
            - 'debt_service' (debt servicing)
            - 'pg' (pasture growth)
            - 'product_price' (product price)
            - 'sup_feed_cost' (supplementary feed cost)
        :param x: x axis to plot against 'time' or a ys variable
        :param sims: sim numbers to plot
        :param mult_as_lines: bool if True, plot multiple sims as lines,
                                   if False, plot multiple sims boxplot style pdf
        :param twin_axs: bool if True, plot each y on twin axis, if False, plot on subplots
        :param start_year: the year to start plotting from (default 2000) only affects the transition from
                           year 1, year2, etc. to datetime
        :return:
        """
        ys = np.atleast_1d(ys)
        assert set(ys).issubset(self.attr_dict.keys()), (f'ys must be one or more of: {self.attr_dict.keys()}, got'
                                                         f'{set(ys) - set(self.attr_dict.keys())}')
        assert x in self.attr_dict.keys() or x == 'time', f'x must be "time" or one of: {self.attr_dict.keys()}, got {x}'
        assert isinstance(mult_as_lines, bool), f'mult_as_lines must be bool, got {type(mult_as_lines)}'
        assert isinstance(twin_axs, bool), f'twin_axs must be bool, got {type(twin_axs)}'
        if twin_axs and len(ys) > 2:
            raise NotImplementedError('twin_axs only implemented for 1 or 2 ys, it gets super messy otherwise')

        if sims is None:
            sims = range(self.nsims)

        sims = np.atleast_1d(sims)
        assert set(sims).issubset(range(self.nsims)), f'sims must be in range({self.nsims}), got {sims}'

        if not mult_as_lines and len(sims) == 1:
            mult_as_lines = True
            warnings.warn('only one sim to plot, setting mult_as_lines=True')

        if len(sims) > 20 and mult_as_lines:
            warnings.warn('more than 20 sims to plot, colors will be reused')
        base_xtime = None
        if x == 'time':
            base_xtime = np.array([datetime.date(
                year=start_year + yr, month=m, day=d
            ) for yr, m, d in zip(self.all_year, self.all_months, self.all_days)])

        # setup plots
        if twin_axs:
            axs = []
            linestyles = ['solid', 'dashed']
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize)
            axs.append(ax)
            if len(ys) > 1:
                for y in ys[1:]:
                    axs.append(ax.twinx())
        else:
            fig, axs = plt.subplots(len(ys), 1, sharex=True, figsize=figsize)
            linestyles = ['solid'] * len(ys)

        ycolors = get_colors(ys)

        # plot data
        for y, ax, ls, yc in zip(ys, axs, linestyles, ycolors):

            if mult_as_lines:
                sim_colors = get_colors(sims)
                for sim, c in zip(sims, sim_colors):
                    if x == 'time':
                        use_x = base_xtime
                        assert use_x is not None, 'base_xtime must be set'
                    else:
                        use_x = getattr(self, self.obj_dict[x])[sim]
                        idx = np.argsort(use_x)
                    ax.plot(use_x, getattr(self, self.obj_dict[y])[:, sim][idx], label=f'sim {sim}')
            else:
                usey = getattr(self, self.obj_dict[y])[:, sims]
                if x == 'time':
                    use_x = base_xtime
                    ax.plot(use_x, np.nanmedian(usey, axis=1), label='median', color=yc)
                    ax.fill_between(use_x, np.nanpercentile(usey, 25, axis=1), np.nanpercentile(usey, 75, axis=1),
                                    label='25-75%', color=yc, alpha=0.25)
                    ax.fill_between(use_x, np.nanpercentile(usey, 5, axis=1), np.nanpercentile(usey, 95, axis=1),
                                    label='5-95%', color=yc, alpha=0.25)

                else:
                    raise NotImplementedError  # todo need to bin x and y

            ax.set_ylabel(self.attr_dict[y])
        ax = axs[-1]
        ax.set_xlabel(x)
        ax.legend()
        return fig, axs

    def calculate_feed_needed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_production(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_next_state(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_sup_feed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_running_cost(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_debt_servicing(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def reset_state(self):
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')


class DummySimpleFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }

    month_reset = 7  # trigger farm reset on day 1 in July

    def calculate_feed_needed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_production(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_next_state(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_sup_feed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_running_cost(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_debt_servicing(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def reset_state(self):
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')


# todo make child class for dairy, dairy support, beef & sheep

def get_colors(vals, cmap_name='tab20'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = get_cmap(cmap_name)
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = get_cmap(cmap_name)
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}
