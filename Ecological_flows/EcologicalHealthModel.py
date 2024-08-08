"""
created matt_dumont 
on: 8/8/24
"""
import tempfile

import numpy as np
import pandas as pd
from pathlib import Path
from project_base import proj_root


class EcoHealthModel:
    _weighting_options = ('pooled', 'duncan_grey', 'rich_allibone', 'greg_burrell', 'adrian_meredith', 'unweighted')
    _baseline_options = ('min_max')
    malf = 42.2007397
    maf = 991.5673849310346
    flow_limit = 50
    temperature_thresholds = (19, 21, 24)
    species_coeffs = {
        "longfin_eel_lt_300": (-9.045618237519400E-09, 3.658952327544510E-06,
                             5.653574369241410E-04, 3.858556802202370E-02,
                             3.239955996233250E-01, 9.987638834796250E+01),
        "torrent_fish": (2.896163694304270E-08, 1.167620629575640E-05,
                         + 1.801041895279500E-03, - 1.329402534268910E-01,
                         + 5.277167341236740E+00, - 1.408366189647840E+01),
        "brown_trout_adult": (4.716969949537670E-09, - 2.076496120868080E-06,
                              + 3.361640291880770E-04, - 2.557607121249140E-02,
                              + 1.060052581008110E+00, + 3.627596900757210E+0),
        "diatoms": (7.415806641571640E-11, - 3.448627575182280E-08,
                    + 6.298888857172090E-06, - 5.672527158325650E-04,
                    + 2.595917911761800E-02, - 1.041530354852930E-01),
        "long_filamentous": (-2.146620894005660E-10, + 8.915219136657130E-08,
                             - 1.409667339556760E-05, + 1.057153790947640E-03,
                             - 3.874332961128240E-02, + 8.884973169426100E-01),

    }
    species_limits = {
        "longfin_eel_lt_300": (18, 130), "torrent_fish": (18, 130),
        "brown_trout_adult": (18, 130), "diatoms": (18, 130), "long_filamentous": (18, 130)}

    def __init__(self, ts_data, flow_data, temp_data, weighting='pooled', baseline='historical', scen_names=None):
        """
        Initialise and run the EcoHealthModel for m flow/temperature scenarios of n days
        :param ts_data: Timedata, must be array like with shape n, and be datetime like and must be either daily data (true time) or daily 365 day year data (e.g. excluding leap days)
        :param flow_data: river flow data (m3/s) of shape (m,n)
        :param temp_data: air temperature data (C) of shape (m,n)
        :param weighting: str, name of the expert weighting to use see self._weighting_options
        :param baseline: str, name of the baseline to use see self._baseline_options
        """
        assert weighting in self._weighting_options, f'weighting must be one of {self._weighting_options}, got {weighting}'
        self.weighting = _get_expert_weightings(weighting)
        self.baseline = _get_baseline_min_max(baseline)
        self._check_set_ts_flow_temp_data(ts_data, flow_data, temp_data, scen_names)
        self.water_year_datasets = {}
        self.scores = {}
        self._calc_stats()
        self._calc_scores()

    def _check_set_ts_flow_temp_data(self, ts_data, flow_data, temp_data, scen_names=None):
        """
        Check the input data is valid
        :param ts_data: Timedata, must be array like with shape (n,) and be datetime like and must be either daily data (true time) or daily 365 day year data (e.g. excluding leap days)
        :param flow_data: river flow data (m3/s) of shape (n,m)
        :param temp_data: air temperature data (C) of shape (n,m)
        """
        assert np.isfinite(flow_data).all(), 'flow_data must have no NaN or inf values'
        assert np.isfinite(temp_data).all(), 'temp_data must have no NaN or inf values'
        flow_data = np.atleast_2d(flow_data)
        temp_data = np.atleast_2d(temp_data)
        assert flow_data.shape == temp_data.shape, (f'flow_data and temp_data must have the same shape, '
                                                    f'got {flow_data.shape=} and {temp_data.shape=}')

        ts_data = pd.to_datetime(ts_data).sort_values()
        assert not ts_data.duplicated().any(), 'ts_data must have no duplicate values'
        expect_ts_data = pd.date_range(ts_data[0], ts_data[-1], freq='D')
        expect_ts_data_no_leap = pd.Series(expect_ts_data)
        expect_ts_data_no_leap = expect_ts_data_no_leap[
            ~((expect_ts_data_no_leap.dt.month == 2) & (expect_ts_data_no_leap.dt.day == 29))]

        problem = True
        if len(ts_data) == len(expect_ts_data_no_leap):
            if (ts_data == expect_ts_data_no_leap).all():
                self.leap_year = False
                problem = False
        elif len(ts_data) == len(expect_ts_data):
            if (ts_data == expect_ts_data).all():
                problem = False
                self.leap_year = True
        if problem:
            missing_days = expect_ts_data_no_leap[~expect_ts_data_no_leap.isin(ts_data)]
            extra_days = ts_data[~ts_data.isin(expect_ts_data)]
            raise ValueError(
                f'ts_data must be daily data with no missing days (except leap days) {missing_days=} {extra_days=}')
        self.ts_data = pd.Series(ts_data)
        self.hydro_year = pd.Series(ts_data.year)
        idx = self.ts_data.dt.month > 6
        self.hydro_year[idx] += 1

        # check that there are full hydro years
        temp = self.hydro_year.groupby(self.hydro_year).count()
        temp.index.name = 'hydro_year'
        temp.name = 'n_days'
        bad = temp[(temp < 365) | (temp > 366)]
        if not bad.empty:
            raise ValueError(f'hydro years must have 365 or 366 days, got {bad}')
        self.unique_hydro_years = self.hydro_year.unique()
        self.ndays = len(ts_data)
        assert flow_data.shape[0] == self.ndays, (f'flow_data must have the same number of days as ts_data, '
                                                  f'got {flow_data.shape[0]=} and {self.ndays=}')
        self.nscenarios = flow_data.shape[1]
        if scen_names is None:
            scen_names = np.arange(self.nscenarios)
            scen_names = pd.Index(scen_names, dtype=int)
        assert len(scen_names) == self.nscenarios, (f'scen_names must have the same number of scenarios as flow_data, '
                                                    f'got {len(scen_names)=} and {self.nscenarios=}')
        self.flow_data = pd.DataFrame(data=flow_data.astype(float), index=self.ts_data, columns=scen_names)
        self.air_temp_data = temp_data.astype(float)

        # water temp regression
        m, b = _get_air_water_temp_reg()
        self.water_temp_data = pd.DataFrame(data=self.air_temp_data * m + b, index=self.ts_data, columns=scen_names)

    def _calc_stats(self):

        day_7_roll = self.flow_data.rolling(window=7, center=True, min_periods=4).mean()
        self.water_year_datasets['alf'] = day_7_roll.groupby(self.hydro_year.values).min()

        temp = self.flow_data.copy()
        temp_cols = temp.columns

        self.water_year_datasets['median_flow'] = temp.groupby(self.hydro_year.values).median()
        self.water_year_datasets['max_flow'] = temp.groupby(self.hydro_year.values).max()

        temp2 = temp.copy()
        temp2 = temp2 < self.malf
        self.water_year_datasets['days_below_malf'] = temp2.groupby(self.hydro_year.values).sum()

        temp2 = temp.copy()
        temp2 = temp2 < self.flow_limit
        self.water_year_datasets['days_below_flow_limit'] = temp2.groupby(self.hydro_year.values).sum()

        temp2 = temp.copy()
        temp2 = temp2 > self.maf
        self.water_year_datasets['days_above_maf'] = temp2.groupby(self.hydro_year.values).sum()

        for t_threshold in self.temperature_thresholds:
            temp2 = self.water_temp_data.copy()
            temp2 = temp2 > t_threshold
            self.water_year_datasets[f'days_above_{t_threshold}C'] = temp2.groupby(self.hydro_year.values).sum()

        # number of consecutive MALF and flow limit exceedances of length days or more.
        for length in [7, 14, 21, 28]:
            def count_periods(x):
                t = pd.DataFrame({'a': np.cumsum(x) * x, 'b': np.cumsum(~x)})
                t = t.groupby('b')['a'].max()
                t = t.loc[t > 0]
                t2 = np.concatenate([[0], t.values])
                t2 = np.diff(t2)
                return np.sum(t2 >= length)

            temp2 = temp.copy()
            temp2 = temp2 < self.malf
            self.water_year_datasets[f'malf_events_{length:02d}d'] = temp2.groupby(self.hydro_year.values).agg(count_periods)

            temp2 = temp.copy()
            temp2 = temp2 < self.flow_limit
            self.water_year_datasets[f'flow_limit_events_{length:02d}d'] = temp2.groupby(self.hydro_year.values).agg(
                count_periods)

        self.water_year_datasets['malf_anomaly'] = self.malf - self.water_year_datasets['alf']

        # Mean annual flood (MAF)
        self.water_year_datasets['maf_anomaly'] = self.water_year_datasets['max_flow'] - self.maf

        self.water_year_datasets['maf_malf_product'] = (
                self.water_year_datasets['days_above_maf'] * self.water_year_datasets['days_below_malf'])

        # implementing the WUA calculation for each species using the ALF
        for species in self.species_coeffs.keys():
            wua = _wua_poly(self.water_year_datasets['alf'], *self.species_coeffs[species])
            minf, maxf = self.species_limits[species]
            idx = (self.water_year_datasets['alf'].values <= minf) | (self.water_year_datasets['alf'].values >= maxf)
            wua[idx] = 0
            self.water_year_datasets[f'{species}_wua'] = wua

        for k, stat in self.water_year_datasets.items():
            assert (stat.index == self.unique_hydro_years).all(), f'{k=} {stat.index=} {self.unique_hydro_years=}'

    def _calc_scores(self):
        # todo I have changed some of the names... make sure this doesn't cause problesm...
        assert set(self.water_year_datasets.keys()).issuperset(self.weighting.keys()), (
            f'missing keys... {set(self.weighting.keys()) - set(self.water_year_datasets.keys())}')
        for var, weight in self.weighting.items():
            is_higher_better = var in ["longfin_eel_lt_300_wua", "torrent_fish_wua", "brown_trout_adult_wua",
                                       "diatoms_wua",
                                       "long_filamentous_wua", "anomaly"]
            min_value = self.baseline[f'{var}_min']
            max_value = self.baseline[f'{var}_max']
            self.scores[var] = score_variable(self.water_year_datasets[var], min_value, max_value, is_higher_better) * \
                               self.weighting[var]

        self.mean_water_year_score = pd.DataFrame(np.concatenate(
            [score.values[np.newaxis] for score in self.scores.values()], axis=0).mean(axis=0),
                                                  index=list(self.scores.values())[0].index,
                                                  columns=list(self.scores.values())[0].columns)
        self.mean_ts_score = self.mean_water_year_score.mean(axis=0)

    def to_hdf(self, path, inc_unique_scores=False, inc_stats=False):
        path = Path(path)
        with pd.HDFStore(path) as store:
            # save scores
            store.put('mean_ts_scores', self.mean_ts_score)
            store.put('mean_water_year_scores', self.mean_water_year_score)
            if inc_unique_scores:
                for key, value in self.scores.items():
                    store.put(key, value)
            if inc_stats:
                for key, value in self.water_year_datasets.items():
                    store.put(key, value)


def _get_expert_weightings(expert_name):
    """
    :param expert_name: str, name of the expert
    :return: dict, weightings for each ecological flow component
    """
    # keynote the median weightings for each variable have been used
    if expert_name == 'duncan_grey':
        return {'longfin_eel_lt_300_wua': 2, 'torrent_fish_wua': 3.5, 'brown_trout_adult_wua': 2.5,
                'diatoms_wua': 2.5,
                'long_filamentous_wua': 4.5,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'malf_anomaly': 4,
                'malf_events_07d': 2.5, 'malf_events_14d': 3.5,
                'malf_events_21d': 4, 'malf_events_28d': 4,
                'flow_limit_events_07d': 2.5, 'flow_limit_events_14d': 2.5,
                'flow_limit_events_21d': 3.5, 'flow_limit_events_28d': 4,
                'days_above_19C': 2.5, 'days_above_21C': 3, 'days_above_24C': 3.5,
                'days_above_maf': 3, 'maf_anomaly': 3, 'maf_malf_product': 3.5}
    elif expert_name == 'rich_allibone':
        return {'longfin_eel_lt_300_wua': 1.5, 'torrent_fish_wua': 3.5, 'brown_trout_adult_wua': 1.5,
                'diatoms_wua': 4,
                'long_filamentous_wua': 0,
                'days_below_malf': 2, 'days_below_flow_limit': 2, 'malf_anomaly': 0,
                'malf_events_07d': 1.5, 'malf_events_14d': 2,
                'malf_events_21d': 3, 'malf_events_28d': 3.5,
                'flow_limit_events_07d': 0.75, 'flow_limit_events_14d': 0.75,
                'flow_limit_events_21d': 0.75, 'flow_limit_events_28d': 1,
                'days_above_19C': 2.5, 'days_above_21C': 3.25, 'days_above_24C': 4.25,
                'days_above_maf': 3, 'maf_anomaly': 0, 'maf_malf_product': 0}
    elif expert_name == 'greg_burrell':
        return {'longfin_eel_lt_300_wua': 1, 'torrent_fish_wua': 1, 'brown_trout_adult_wua': 1,
                'diatoms_wua': 1,
                'long_filamentous_wua': -1,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'malf_anomaly': 4,
                'malf_events_07d': 3.5, 'malf_events_14d': 3.5,
                'malf_events_21d': 3.5, 'malf_events_28d': 3.5,
                'flow_limit_events_07d': 3.5, 'flow_limit_events_14d': 3.5,
                'flow_limit_events_21d': 3.5, 'flow_limit_events_28d': 3.5,
                'days_above_19C': 4, 'days_above_21C': 4, 'days_above_24C': 4,
                'days_above_maf': 4, 'maf_anomaly': 4, 'maf_malf_product': 4}
    elif expert_name == 'adrian_meredith':
        return {'longfin_eel_lt_300_wua': 3.5, 'torrent_fish_wua': 4.5, 'brown_trout_adult_wua': 3.5,
                'diatoms_wua': 4,
                'long_filamentous_wua': -3,
                'days_below_malf': 4, 'days_below_flow_limit': 4.5, 'malf_anomaly': 2,
                'malf_events_07d': 4.5, 'malf_events_14d': 4,
                'malf_events_21d': 4.5, 'malf_events_28d': 5,
                'flow_limit_events_07d': 4.5, 'flow_limit_events_14d': 4,
                'flow_limit_events_21d': 4.5, 'flow_limit_events_28d': 4,
                'days_above_19C': 3, 'days_above_21C': 4, 'days_above_24C': 4,
                'days_above_maf': 2, 'maf_anomaly': 3, 'maf_malf_product': 1.5}
    elif expert_name == 'pooled':
        return {'longfin_eel_lt_300_wua': 2, 'torrent_fish_wua': 3.125, 'brown_trout_adult_wua': 2.125,
                'diatoms_wua': 2.875,
                'long_filamentous_wua': 0.125,
                'days_below_malf': 3.5, 'days_below_flow_limit': 3.375, 'malf_anomaly': 2.5,
                'malf_events_07d': 3, 'malf_events_14d': 3.25,
                'malf_events_21d': 3.75, 'malf_events_28d': 4,
                'flow_limit_events_07d': 2.8125, 'flow_limit_events_14d': 2.6875,
                'flow_limit_events_21d': 3.0625, 'flow_limit_events_28d': 3.125,
                'days_above_19C': 2.75, 'days_above_21C': 3.5625, 'days_above_24C': 3.9375,
                'days_above_maf': 3, 'maf_anomaly': 2.5, 'maf_malf_product': 2.25}
    elif expert_name == 'unweighted':
        return {'longfin_eel_lt_300_wua': 1, 'torrent_fish_wua': 1, 'brown_trout_adult_wua': 1,
                'diatoms_wua': 1,
                'long_filamentous_wua': 1,
                'days_below_malf': 1, 'days_below_flow_limit': 1, 'malf_anomaly': 1,
                'malf_events_07d': 1, 'malf_events_14d': 1,
                'malf_events_21d': 1, 'malf_events_28d': 1,
                'flow_limit_events_07d': 1, 'flow_limit_events_14d': 1,
                'flow_limit_events_21d': 1, 'flow_limit_events_28d': 1,
                'days_above_19C': 1, 'days_above_21C': 1,
                'days_above_24C': 1,
                'days_above_maf': 1, 'maf_anomaly': 1, 'maf_malf_product': 1}
    else:
        raise ValueError(f'invalid expert name {expert_name}')


def _get_baseline_min_max(baseline='historical'):
    """
    get the baseline min and max values for each ecological flow component
    :param baseline: historical (min and max values from the full historical record)
    :return:
    """
    if baseline == 'historical':
        baseline_data = {'malf': 42.2007397, 'maf': 991.5673849310346, "longfin_eel_lt_300_wua_max": 146,
                         "torrent_fish_wua_max": 71,
                         "brown_trout_adult_wua_max": 19, "diatoms_wua_max": 0.28,
                         "long_filamentous_wua_max": 0.31, "longfin_eel_lt_300_wua_min": 426, "torrent_fish_wua_min": 395,
                         "brown_trout_adult_wua_min": 25, "diatoms_wua_min": 0.38, "long_filamentous_wua_min": 0.39,
                         'days_below_malf_min': 0, 'days_below_malf_max': 70,
                         'days_below_flow_limit_min': 0, 'days_below_flow_limit_max': 108, 'malf_anomaly_min': -18.20,
                         'malf_anomaly_max': 16.15, 'malf_events_07d_min': 0, 'malf_events_07d_max': 4,
                         'malf_events_14d_min': 0, 'malf_events_14d_max': 2,
                         'malf_events_21d_min': 0, 'malf_events_21d_max': 1,
                         'malf_events_28d_min': 0, 'malf_events_28d_max': 1,
                         'flow_limit_events_07d_min': 0, 'flow_limit_events_07d_max': 6,
                         'flow_limit_events_14d_min': 0, 'flow_limit_events_14d_max': 4,
                         'flow_limit_events_21d_min': 0, 'flow_limit_events_21d_max': 2,
                         'flow_limit_events_28d_min': 0, 'flow_limit_events_28d_max': 1, 'days_above_19C_min': 0,
                         'days_above_19C_max': 23,
                         'days_above_21C_min': 0, 'days_above_21C_max': 3,
                         'days_above_24C_min': 0, 'days_above_24C_max': 1, 'maf_anomaly_min': -977.0379620689654,
                         'maf_anomaly_max': 431.5956442310345,
                         'days_above_maf_min': 0, 'days_above_maf_max': 3, 'maf_malf_product_min': 0,
                         'maf_malf_product_max': 180}
    else:
        raise ValueError(f'invalid baseline {baseline}')
    return baseline_data


def _get_air_water_temp_reg(recalc=False):  # todo this is to mean water temperature, but need maximum, need to clean up regression
    # todo a regression approach is probably not great... we also need to consider the radient forcing, and the historisis...
    # todo lit review...
    """
    get the regression between air and water temperature

    :param recalc:
    :return:
    """
    data_path = proj_root.joinpath('Ecological_flows', 'temperature_regressor_data.npz')
    if data_path.exists() and not recalc:
        data = np.load(data_path)
        m = data['m'][0]
        b = data['b'][0]
    else:
        from komanawa.kslcore import KslEnv
        from sklearn.linear_model import LinearRegression
        water_base_path = KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'temp_data',
                                                                           'Waiau_daily_mean.csv')
        base_path = KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'temp_data', 'waiau_temp',
                                                                     'daily_temperature_Cheviot Ews_data.csv')
        air_temp_df = pd.read_csv(base_path)
        air_temp_df = air_temp_df.rename(columns={'time': 'date'})
        air_temp_df['date'] = pd.to_datetime(air_temp_df['date'], format='mixed')

        daily_water_temp = pd.read_csv(water_base_path)
        daily_water_temp['Date & Time'] = pd.to_datetime(daily_water_temp['Date & Time'], format='mixed')
        daily_water_temp = daily_water_temp.rename(
            columns={'Date & Time': 'date', 'Water Temp (degC)': 'daily_mean_water_temp'})
        merged_df = daily_water_temp.merge(air_temp_df)
        merged_df.to_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'temp_data', 'waiau_temp',
                                                                          'air_water_temp_data.csv'))
        regression_df = merged_df.dropna()
        x = regression_df.loc[:, 'daily_mean_air_temp'].values.reshape(-1, 1)
        y = regression_df.loc[:, 'daily_mean_water_temp'].values.reshape(-1, 1)

        temp_regr = LinearRegression()
        temp_regr.fit(x, y)
        np.savez_compressed(data_path, m=[temp_regr.coef_], b=[temp_regr.intercept_], r2=temp_regr.score(x, y), air_t=x,
                            water_t=y)
        m = temp_regr.coef_
        b = temp_regr.intercept_
    return m, b


def _wua_poly(x, a, b, c, d, e, f):
    """a function that reads in coefficients and returns a polynomial with the coeffs
    inserted"""
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def score_variable(value: float, min_value: float, max_value: float, is_higher_better: bool) -> float:
    # This function calculates the score for a single variable
    if min_value == max_value:
        return 0
    score = (value - min_value) / (max_value - min_value)
    if not is_higher_better:
        score = (score * 2) - 1
        score = np.round((score * -3) * 2.0) / 2.0
    else:
        score = (score * 2) - 1
        score = np.round((score * 3) * 2.0) / 2.0
    return score


if __name__ == '__main__':
    # Example usage
    # todo test with a longer record, multiple scenarios.
    # todo check against the old model... getting differences...
    from komanawa.kslcore import KslEnv
    import time

    flow_data = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info', '2024_test_data',
                                                         'measured_flow_data_storyline_data_2bad_2024_test.csv'))
    temp_data = pd.read_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info',
                                                                             '2024_test_data',
                                                                             'temperature_data_storylines_2024_test.csv'))
    temp_data['date'] = pd.to_datetime(temp_data['datetime'], format='%d/%m/%Y')
    temp_data = temp_data.loc[temp_data['date'] >= '2018-07-01']
    flow_data['date'] = pd.to_datetime(flow_data['datetime'], format='%d/%m/%Y')
    flow_data = flow_data.loc[flow_data['date'] >= '2018-07-01']
    t = time.time()
    test = EcoHealthModel(flow_data['date'].values, flow_data['flow'].values[:, np.newaxis],
                          temp_data['temp'].values[:, np.newaxis],
                          weighting='pooled', baseline='historical')
    print('took', time.time() - t, 's')
    with tempfile.TemporaryDirectory() as tdir:
        test.to_hdf(Path(tdir).joinpath('test.hdf'), inc_stats=True, inc_unique_scores=True)
    pass
    print(test.mean_ts_score)
