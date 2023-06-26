"""
created matt_dumont 
on: 26/06/23
"""
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import netCDF4 as nc


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
    calculate_feed_needed = None  # dummy value, must be set in child class
    calculate_production = None  # dummy value, must be set in child class
    month_reset = None  # dummy value, must be set in child class
    calculate_next_state = None  # dummy value, must be set in child class
    calculate_sup_feed = None  # dummy value, must be set in child class
    reset_state = None  # dummy value, must be set in child class
    calculate_running_cost = None  # dummy value, must be set in child class
    calculate_debt_servicing = None  # dummy value, must be set in child class

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price):
        """

        :param all_months: integer months, defines time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (time_len,) or (time_len, nsims)
        :param ifeed: initial feed number or np.ndarray shape (nsims,)
        :param imoney: initial money number or np.ndarray shape (nsims,)
        :param sup_feed_cost: cost of supplementary feed $/kgDM or np.ndarray shape (nsims,) or (time_len, nsims)
        :param prod_price: income price $/kg product or np.ndarray shape (nsims,) or (time_len, nsims)
        """
        # define model shape
        assert set(all_months).issubset(set(range(1, 13))), f'months must be in range 1-12'
        self.time_len = len(all_months)
        self.nsims = len(istate)
        self.model_shape = (self.time_len + 1, self.nsims)
        self._run = False

        assert all(len(e) == self.nsims for e in (ifeed, imoney))
        all_months = np.atleast_1d(all_months)
        ifeed = np.atleast_1d(ifeed)
        imoney = np.atleast_1d(imoney)

        # handle pg input options
        pg = np.atleast_1d(pg)
        if pg.ndim == 1:
            assert len(pg) == self.time_len, f'pg must be length time: {self.time_len=}, got: {len(pg)=}'
            pg = np.repeat(pg[:, np.newaxis], self.nsims, axis=1)
        assert pg.shape == self.model_shape, f'pg shape must be: {self.model_shape=}, got: {pg.shape=}'
        self.pg = np.concatenate([pg[[0]], pg])

        # handle sup_feed_cost input options
        if pd.api.types.is_number(sup_feed_cost):
            sup_feed_cost = np.full(self.model_shape, sup_feed_cost)
        elif sup_feed_cost.ndim == 1:
            assert len(
                sup_feed_cost) == self.time_len, f'sup_feed_cost must be length time: {self.time_len=}, got: {len(sup_feed_cost)=}'
            sup_feed_cost = np.repeat(sup_feed_cost[:, np.newaxis], self.nsims, axis=1)
        else:
            assert sup_feed_cost.shape == self.model_shape, f'sup_feed_cost shape must be: {self.model_shape=}, got: {sup_feed_cost.shape=}'

        self.sup_feed_cost = np.concatenate([sup_feed_cost[[0]], sup_feed_cost], axis=0)

        # manage product price options
        if pd.api.types.is_number(product_price):
            product_price = np.full(self.model_shape, product_price)
        elif product_price.ndim == 1:
            assert len(
                product_price) == self.time_len, f'product_price must be length time: {self.time_len=}, got: {len(product_price)=}'
            product_price = np.repeat(product_price[:, np.newaxis], self.nsims, axis=1)
        else:
            assert product_price.shape == self.model_shape, f'product_price shape must be: {self.model_shape=}, got: {product_price.shape=}'

        self.product_price = np.concatenate([product_price[[0]], product_price], axis=0)

        # assertion errors for system specific values
        assert not any([e is None for e in [
            self.states,
            self.month_reset,
            self.calculate_feed_needed,
            self.calculate_production,
            self.calculate_next_state,
            self.calculate_sup_feed,
            self.calculate_running_cost,
            self.calculate_debt_servicing,
            self.reset_state,
        ]]), 'must set all system specific values in child class'

        # todo check the following are methods and have correct kwargs
        check_funcs = (
            # kwargs of month, current_state
            self.calculate_feed_needed,
            self.calculate_production,
            self.calculate_next_state,
            self.calculate_sup_feed,
            self.calculate_running_cost,
            self.calculate_debt_servicing,

            # no kwargs
            self.reset_state,
        )

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

        # # todo internal check of shapes
        # ntime shapes

        raise NotImplementedError

    def run_model(self):
        """
        run the model
        :return:
        """
        assert not self._run, f'model has already been run'
        for i_month in range(1, self.time_len + 1):
            month = self.all_months[i_month]

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
            if month == self.month_reset:
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

            ds.createVariable('month', 'i4', ('time',))
            ds.createVariable('sim', 'i4', ('nsims',))
            ds.variables['month'][:] = self.all_months
            ds.variables['month'].setncattr('long_name', 'month of the year')
            ds.createVariable('year', 'i4', ('nsims',))
            ds.variables['year'].setncattr('long_name', 'year')
            ds.variables['year'][:] = np.cumsum(self.all_months == self.month_reset) + 1,

            ds.variables['sims'][:] = range(self.nsims)
            ds.variables['sims'].setncattr('long_name', 'simulation number')

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
                       product_price=product_price[1:])
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

    def plot_results(self):  # todo
        raise NotImplementedError('plotting not implemented yet')


# todo make child class for dairy, dairy support, beef & sheep