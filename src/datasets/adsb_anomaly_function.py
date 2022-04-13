import numpy as np
import pandas as pd
from .adsb_dataset import ADSBDataset


class ADSBAnomalyFunction:
    # ----- Functions generating the anomalous dimension --------- #
    # A MultivariateAnomalyFunction should return a tuple containing the following three values:
    # * The values of the second dimension (array of max `interval_length` numbers)
    # * Starting point for the anomaly
    # * End point for the anomaly section
    # The last two values are ignored for generation of not anomalous data

    # Get a dataset by passing the method name as string. All following parameters
    # are passed through. Throws AttributeError if attribute was not found.
    @staticmethod
    def get_multivariate_dataset(method, name=None, group_size=None, x_train=None, x_test=None, y_train=None,
                                 *args, **kwargs):
        name = name or f'ADS-B {method} Curve Outliers'
        func = getattr(ADSBAnomalyFunction, method)
        return ADSBDataset(anomaly_func=func, name=name, group_size=group_size, x_train=x_train, x_test=x_test,
                           y_train=y_train, *args, **kwargs)

    @staticmethod
    def doubled(curve_values, anomalous, _):
        factor = 4 if anomalous else 2
        return curve_values * factor, 0, len(curve_values)

    @staticmethod
    def inversed(curve_values, anomalous, _):
        factor = -2 if anomalous else 2
        return curve_values * factor, 0, len(curve_values)

    @staticmethod
    def shrinked(curve_values, anomalous, _):
        if not anomalous:
            return curve_values, -1, -1
        else:
            new_curve = curve_values[::2]
            nonce = np.zeros(len(curve_values) - len(new_curve))
            values = np.concatenate([nonce, new_curve])
            return values, 0, len(values)

    @staticmethod
    def xor(curve_values, anomalous, interval_length):
        pause_length = interval_length - len(curve_values)
        if not anomalous:
            # No curve during the other curve in the 1st dimension
            nonce = np.zeros(len(curve_values))
            # Insert a curve with the same amplitude during the pause of the 1st dimension
            new_curve = ADSBAnomalyFunction.shrink_curve(curve_values, pause_length)
            return np.concatenate([nonce, new_curve]), -1, -1
        else:
            # Anomaly: curves overlap (at the same time or at least half overlapping)
            max_pause = min(len(curve_values) // 2, pause_length)
            nonce = np.zeros(max_pause)
            return np.concatenate([nonce, curve_values]), len(nonce), len(curve_values)

    @staticmethod
    def delayed(curve_values, anomalous, interval_length):
        if not anomalous:
            return curve_values, -1, -1
        else:
            # The curve in the second dimension occurs a few timestamps later
            left_space = interval_length - len(curve_values)
            delay = min(len(curve_values) // 2, left_space)
            nonce = np.zeros(delay)
            values = np.concatenate([nonce, curve_values])
            return values, 0, len(values)

    @staticmethod
    def delayed_missing(curve_values, anomalous, interval_length):
        starting_point = len(curve_values) // 5
        # If the space is too small for the normal curve we're shrinking it (which is not anomalous)
        left_space = interval_length - starting_point
        new_curve_length = min(left_space, len(curve_values))
        if not anomalous:
            # The curve in the second dimension occurs a few timestamps later
            nonce = np.zeros(starting_point)
            new_curve = ADSBAnomalyFunction.shrink_curve(curve_values, new_curve_length)
            values = np.concatenate([nonce, new_curve])
            return values, -1, -1
        else:
            end_point = starting_point + new_curve_length
            nonce = np.zeros(end_point)
            return nonce, starting_point, end_point

    """
        This is a helper function for shrinking an already generated curve.
    """

    @staticmethod
    def shrink_curve(curve_values, new_length):
        if new_length == len(curve_values):
            return curve_values
        orig_amplitude = max(abs(curve_values))
        orig_amplitude *= np.sign(curve_values.mean())
        return ADSBDataset.get_curve(new_length, orig_amplitude)

    @staticmethod
    def random(x_test, anomaly_rate):
        test_number = x_test.shape[0]
        x_test = x_test.to_numpy()
        anomaly_number = int(anomaly_rate * test_number)
        y_test = np.zeros(test_number, dtype=bool)
        anomaly_time = np.random.choice(test_number, size=anomaly_number, replace=False)
        y_test[anomaly_time] = 1
        for idx in anomaly_time:
            anomaly_dim_number =np.random.randint(1, 6)
            anomaly_dim = np.random.choice(5, size=anomaly_dim_number, replace=False)
            x_test[idx, anomaly_dim] = x_test[idx, anomaly_dim] * 3
        return pd.DataFrame(x_test), pd.DataFrame(y_test)

    @staticmethod
    def heading(x_test, continuous, number, length, anomaly_rate, intensity):
        test_number = x_test.shape[0]
        x_test = x_test.to_numpy()
        y_test = np.zeros(test_number, dtype=bool)
        if not continuous:
            anomaly_number = int(anomaly_rate * test_number)
            anomaly_time = np.random.choice(test_number, size=anomaly_number, replace=False)
            y_test[anomaly_time] = 1
            for idx in anomaly_time:
                anomaly_dim = 4
                x_test[idx, anomaly_dim] = x_test[idx, anomaly_dim] * 3
        else:
            start_anomaly_time = np.random.choice(test_number, size=number, replace=False)
            max_intensity = intensity + intensity * (length - 1)
            anomaly = np.arange(intensity, max_intensity + intensity, intensity)
            for s in start_anomaly_time:
                x_test[s:s + length, 4] = x_test[s:s + length, 4] + anomaly
                y_test[s:s + length] = 1
        return pd.DataFrame(x_test), pd.DataFrame(y_test)

    @staticmethod
    def speed(x_test, continuous, number, length, anomaly_rate, intensity):
        test_number = x_test.shape[0]
        x_test = x_test.to_numpy()
        y_test = np.zeros(test_number, dtype=bool)
        if not continuous:
            anomaly_number = int(anomaly_rate * test_number)
            anomaly_time = np.random.choice(test_number, size=anomaly_number, replace=False)
            y_test[anomaly_time] = 1
            for idx in anomaly_time:
                anomaly_dim = 3
                x_test[idx, anomaly_dim] = x_test[idx, anomaly_dim] * 3
        else:
            start_anomaly_time = np.random.choice(test_number, size=number, replace=False)
            max_intensity = intensity + intensity * (length - 1)
            anomaly = np.arange(intensity, max_intensity + intensity, intensity)
            for s in start_anomaly_time:
                x_test[s:s + length, 3] = x_test[s:s + length, 3] + anomaly
                y_test[s:s + length] = 1
        return pd.DataFrame(x_test), pd.DataFrame(y_test)
    @staticmethod
    def altitude(x_test, continuous, number, length, anomaly_rate, intensity):
        test_number = x_test.shape[0]
        x_test = x_test.to_numpy()
        y_test = np.zeros(test_number, dtype=bool)
        if not continuous:
            anomaly_number = int(anomaly_rate * test_number)
            anomaly_time = np.random.choice(test_number, size=anomaly_number, replace=False)
            y_test[anomaly_time] = 1
            for idx in anomaly_time:
                anomaly_dim = 2
                x_test[idx, anomaly_dim] = x_test[idx, anomaly_dim] * 3
        else:
            start_anomaly_time = np.random.choice(test_number, size=number, replace=False)
            max_intensity = intensity + intensity * (length - 1)
            anomaly = np.arange(intensity, max_intensity + intensity, intensity)
            for s in start_anomaly_time:
                x_test[s:s + length, 2] = x_test[s:s + length, 2] + anomaly
                y_test[s:s + length] = 1
        return pd.DataFrame(x_test), pd.DataFrame(y_test)

    @staticmethod
    def ddos(x_test, continuous, number, length, anomaly_rate, neighbor_size):
        test_number = x_test.shape[0]
        x_test = x_test.to_numpy()
        y_test = np.zeros(test_number, dtype=bool)
        if not continuous:
            anomaly_number = int(anomaly_rate * test_number)
            anomaly_time = np.random.choice(test_number, size=anomaly_number, replace=False)
            for idx in anomaly_time:
                x_test[idx-(neighbor_size/2):idx, :] = 0.0
                y_test[idx-(neighbor_size/2):idx] = 1
                x_test[idx+1:idx+(neighbor_size/2), :] = 0.0
                y_test[idx+1:idx+(neighbor_size/2)] = 1
        else:
            start_anomaly_time = np.random.choice(test_number, size=number, replace=False)
            for s in start_anomaly_time:
                e = s + neighbor_size * (length - 1)
                anomaly_time = np.arange(s, e + neighbor_size, neighbor_size)
                for idx in anomaly_time:
                    x_test[idx + 1:idx + neighbor_size, :] = 0.0
                    y_test[idx + 1:idx + neighbor_size] = 1
        return pd.DataFrame(x_test), pd.DataFrame(y_test)

    @staticmethod
    def ghost():
        pass


