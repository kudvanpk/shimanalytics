# For now freely borrowed from skyline, later will clean up as we learn more
import sys
import pandas
import numpy as np
import scipy
import statsmodels.api as sm
import traceback
import logging
from time import time

"""
This is no man's land. Do anything you want in here,
as long as you return a boolean that determines whether the input
timeseries is anomalous or not.

To add an algorithm, define it here, and add its name to settings.ALGORITHMS.
"""
class AnomalySingleTimeSeries():
    """ 
       Takes a single time series and finds anomalies 
       using one or more methods as given by anomaly_type
       anomaly_type can be: ensemble (meaning all), or 
       any one particular type. Currently this package expects
       Expects a series of type where first element is ts in time units (default seconds) 
       relative to now and second element is measurement. Assuming reformatting is done in 
       experiment code with appropriate assumptions
       THIS IS NOT THE MOST EFFICIENT. NEED TO CHANGE THIS TO WORK DIRECTLY ON PANDAS DATA STRUCTURES
       [...........[-22.0, 86378], [-21.0, 86379], [-20.0, 86380], [-19.0, 86381], [-18.0, 86382], [-17.0, 86383], [-16.0, 86384], [-15.0, 86385], [-14.0, 86386], [-13.0, 86387], [-12.0, 86388], [-11.0, 86389], [-10.0, 86390], [-9.0, 86391], [-8.0, 86392], [-7.0, 86393], [-6.0, 86394], [-5.0, 86395], [-4.0, 86396], [-3.0, 86397], [-2.0, 86398], [-1.0, 86399], [0.0, 86400]]
    """ 
    def __init__(self,series, time_unit):
       self.series = series
       self.time_unit = time_unit
       self.ALGORITHMS = [
           'first_hour_average',
           'mean_subtraction_cumulation',
           'stddev_from_average',
           'stddev_from_moving_average',
           'least_squares',
           'grubbs',
           'histogram_bins',
           'median_absolute_deviation',
#           'ks_test',
        ]

    def is_valid_anomaly_type(self, anomaly_type):
       if anomaly_type in self.ALGORITHMS or anomaly_type == 'ensemble':
           return True
       else:
           return False    

    def process(self, anomaly_type):
       ensemble_res = []
       if self.is_valid_anomaly_type(anomaly_type):
          self.anomaly_type = anomaly_type
       else:
          raise 

       if self.anomaly_type == 'ensemble':
          for algorithm in self.ALGORITHMS:
              algorithmfunc = getattr(self, algorithm)
              ensemble_res.append(algorithmfunc())
       else:  
          algorithmfunc = getattr(self, anomaly_type)
          ensemble_res.append(algorithmfunc())
       return ensemble_res
 
    def tail_avg(self):
       """
       This is a utility function used to calculate the average of the last three
       datapoints in the series as a measure, instead of just the last datapoint.
       It reduces noise, but it also reduces sensitivity and increases the delay
       to detection.
       """
       timeseries = self.series
       try:
            t = (timeseries[-1][1] + timeseries[-2][1] + timeseries[-3][1]) / 3
            return t
       except IndexError:
            return timeseries[-1][1]


    def median_absolute_deviation(self):
        """
        A timeseries is anomalous if the deviation of its latest datapoint with
        respect to the median is X times larger than the median of deviations.
        """
        timeseries = self.series
        series = pandas.Series([x[1] for x in timeseries])
        median = series.median()
        demedianed = np.abs(series - median)
        median_deviation = demedianed.median()

        # The test statistic is infinite when the median is zero,
        # so it becomes super sensitive. We play it safe and skip when this happens.
        if median_deviation == 0:
            return False

        test_statistic = demedianed.iget(-1) / median_deviation

        # Completely arbitary...triggers if the median deviation is
        # 6 times bigger than the median
        if test_statistic > 6:
            return True
        return False

    def grubbs(self):
        """
        A timeseries is anomalous if the Z score is greater than the Grubb's score.
        """
        timeseries = self.series
        series = scipy.array([x[1] for x in timeseries])
        stdDev = scipy.std(series)
        mean = np.mean(series)
        tail_average = self.tail_avg()
        z_score = (tail_average - mean) / stdDev
        len_series = len(series)
        threshold = scipy.stats.t.isf(.05 / (2 * len_series), len_series - 2)
        threshold_squared = threshold * threshold
        grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

        return z_score > grubbs_score


    def first_hour_average(self):
        """
        Calcuate the simple average over one hour, FULL_DURATION seconds ago.
        A timeseries is anomalous if the average of the last three datapoints
        are outside of three standard deviations of this value.
        """
        timeseries = self.series
        if self.time_unit == 'seconds':
          FULL_DURATION = 7200
          last_hour_threshold = time() - (FULL_DURATION - 3600)
        if self.time_unit == 'hours':
          FULL_DURATION = 60
          last_hour_threshold = time() - (FULL_DURATION - 30)
        if self.time_unit == 'days':
          FULL_DURATION = 12
          last_hour_threshold = time() - (FULL_DURATION - 6)
        series = pandas.Series([x[1] for x in timeseries if x[0] < last_hour_threshold])
        mean = (series).mean()
        stdDev = (series).std()
        t = self.tail_avg()
 
        return abs(t - mean) > 3 * stdDev


    def stddev_from_average(self):
        """
        A timeseries is anomalous if the absolute value of the average of the latest
        three datapoint minus the moving average is greater than three standard
        deviations of the average. This does not exponentially weight the MA and so
        is better for detecting anomalies with respect to the entire series.
        """
        timeseries = self.series
        series = pandas.Series([x[1] for x in timeseries])
        mean = series.mean()
        stdDev = series.std()
        t = self.tail_avg()

        return abs(t - mean) > 3 * stdDev


    def stddev_from_moving_average(self):
        """
        A timeseries is anomalous if the absolute value of the average of the latest
        three datapoint minus the moving average is greater than three standard
        deviations of the moving average. This is better for finding anomalies with
        respect to the short term trends.
        """
        timeseries = self.series
        series = pandas.Series([x[1] for x in timeseries])
        expAverage = pandas.stats.moments.ewma(series, com=50)
        stdDev = pandas.stats.moments.ewmstd(series, com=50)

        return abs(series.iget(-1) - expAverage.iget(-1)) > 3 * stdDev.iget(-1)


    def mean_subtraction_cumulation(self):
        """
        A timeseries is anomalous if the value of the next datapoint in the
        series is farther than three standard deviations out in cumulative terms
        after subtracting the mean from each data point.
        """
        timeseries = self.series
        series = pandas.Series([x[1] if x[1] else 0 for x in timeseries])
        series = series - series[0:len(series) - 1].mean()
        stdDev = series[0:len(series) - 1].std()
        expAverage = pandas.stats.moments.ewma(series, com=15)

        return abs(series.iget(-1)) > 3 * stdDev


    def least_squares(self):
        """
        A timeseries is anomalous if the average of the last three datapoints
        on a projected least squares model is greater than three sigma.
        """
        timeseries = self.series
        x = np.array([t[0] for t in timeseries])
        y = np.array([t[1] for t in timeseries])
        A = np.vstack([x, np.ones(len(x))]).T
        results = np.linalg.lstsq(A, y)
        residual = results[1]
        m, c = np.linalg.lstsq(A, y)[0]
        errors = []
        for i, value in enumerate(y):
            projected = m * x[i] + c
            error = value - projected
            errors.append(error)

        if len(errors) < 3:
            return False

        std_dev = scipy.std(errors)
        t = (errors[-1] + errors[-2] + errors[-3]) / 3

        return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


    def histogram_bins(self):
        """
        A timeseries is anomalous if the average of the last three datapoints falls
        into a histogram bin with less than 20 other datapoints (you'll need to tweak
        that number depending on your data)

        Returns: the size of the bin which contains the tail_avg. Smaller bin size
        means more anomalous.
        """

        timeseries = self.series
        series = scipy.array([x[1] for x in timeseries])
        t = self.tail_avg()
        h = np.histogram(series, bins=15)
        bins = h[1]
        for index, bin_size in enumerate(h[0]):
            if bin_size <= 20:
                # Is it in the first bin?
                if index == 0:
                    if t <= bins[0]:
                        return True
                # Is it in the current bin?
                elif t >= bins[index] and t < bins[index + 1]:
                        return True

        return False


    def ks_test(self):
        """
        A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
        that data distribution for last 10 minutes is different from last hour.
        It produces false positives on non-stationary series so Augmented
        Dickey-Fuller test applied to check for stationarity.
        """
        timeseries = self.series
        if self.time_unit == 'seconds':
            hour_ago = time() - 3600
            ten_minutes_ago = time() - 600
            reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
            probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])
        if self.time_unit == 'days':
            ten_days_ago = time() - 10
            two_days_ago = time() - 2
            reference = scipy.array([x[1] for x in timeseries if x[0] >= ten_days_ago and x[0] < two_days_ago])
            probe = scipy.array([x[1] for x in timeseries if x[0] >= two_days_ago])


        if reference.size < 20 or probe.size < 20:
            return False

        ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

        if ks_p_value < 0.05 and ks_d > 0.5:
            adf = sm.tsa.stattools.adfuller(reference, 10)
            if adf[1] < 0.05:
                return True

        return False


    
