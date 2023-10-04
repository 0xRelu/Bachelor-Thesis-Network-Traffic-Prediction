import numpy as np
import torch


class StandardScaler:

    def __init__(self, mean=None, std=None, zero_element=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.zero_element = zero_element

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
        self.zero_element = self.transform(0)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


class StandardScalerNp:
    def __init__(self, mean=None, std=None, zero_element=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.zero_element = zero_element

    def fit(self, values):
        # dims = list(range(values.ndim() - 1))
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.zero_element = self.transform(0)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return (values * (self.std + self.epsilon)) + self.mean


class MinMaxScalerNp:
    def __init__(self, min=None, max=None, zero_element=None, epsilon=1e-7):
        self.min = min
        self.max = max
        self.epsilon = epsilon
        self.zero_element = zero_element

    def fit(self, values: np.ndarray):
        # dims = list(range(values.ndim() - 1))
        self.min = np.min(values)
        self.max = np.max(values)
        self.zero_element = self.transform(0)

    def transform(self, values):
        return (values - self.min) / (self.max - self.min + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return (values * (self.max - self.min + self.epsilon)) + self.min


class LogScalerNp:
    def fit(self, values: np.ndarray):
        pass

    def transform(self, values):
        constant = 1.0
        return np.log(values + 1.0)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return np.exp(values)


class RobustScalerNp:
    def __init__(self, mean=None, quantile_75=None, quantile_25=None):
        self.mean = mean
        self.quantile_75 = quantile_75
        self.quantile_25 = quantile_25

    def fit(self, values: np.ndarray):
        self.mean = np.mean(values)
        self.quantile_75 = np.percentile(values, 75)
        self.quantile_25 = np.percentile(values, 25)

    def transform(self, values: np.ndarray):
        return (values - self.mean) / (self.quantile_75 - self.quantile_25)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.quantile_75 - self.quantile_25) + self.mean
