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


class TrafficScalerGlobalEvenNp:
    def __init__(self):
        self.scaler = StandardScalerNp()

    def fit(self, values):
        values_n = np.concatenate(values, axis=0)
        self.scaler.fit(values_n)

    def transform(self, values):
        transformed_values = []

        for i in range(len(values)):
            transformed_values.append(self.scaler.transform(values[i]))

        return transformed_values

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def zero_element(self):
        return self.scaler.zero_element


class TrafficScalerLocalEvenNp:  # TODO check how to do this!!!!!
    def __init__(self):
        self.scaler = []

    def fit(self, values: np.ndarray):
        self.scaler = [StandardScalerNp() for _ in range(len(values))]

        for i in range(len(self.scaler)):
            self.scaler[i].fit(values[i])  # values = [time, size, direction] -> we only want to normalize size

    def transform(self, values: np.ndarray) -> list[np.ndarray]:
        if len(self.scaler) != len(values):
            raise AttributeError("Not same dimension")

        transformed_values = []

        for i in range(len(values)):
            transformed_values.append(self.scaler[i].transform(values[i]))

        return transformed_values

    def fit_transform(self, values: np.ndarray) -> list[np.ndarray]:
        self.fit(values)
        return self.transform(values)

    def zero_element(self, index: int) -> int:
        return self.scaler[index].zero_element


class TrafficScalerLocalSingleNp:
    def __init__(self):
        self.scaler = []

    def fit(self, values: list[np.ndarray]):
        self.scaler = [StandardScalerNp() for _ in range(len(values))]

        for i in range(len(self.scaler)):
            self.scaler[i].fit(values[i][:, 1])  # values = [time, size, direction] -> we only want to normalize size

    def transform(self, values: list[np.ndarray]) -> list[np.ndarray]:
        if len(self.scaler) != len(values):
            raise AttributeError("Not same dimension")

        transformed_values = []

        for i in range(len(values)):
            values_elem: np.ndarray = values[i]
            values_elem[:, 1] = self.scaler[i].transform(values_elem[:, 1])
            transformed_values.append(values_elem)

        return transformed_values

    def fit_transform(self, values: list[np.ndarray]) -> list[np.ndarray]:
        self.fit(values)
        return self.transform(values)

    def zero_element(self, index: int) -> int:
        return self.scaler[index].zero_element


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


class MinMaxScalerNp:
    def __init__(self, min=None, max=None, zero_element=None, epsilon=1e-7):
        self.min = min
        self.max = max
        self.epsilon = epsilon
        self.zero_element = zero_element

    def fit(self, values: np.ndarray):
        # dims = list(range(values.ndim() - 1))
        self.min = np.min(values)
        self.max = np.min(values)
        self.zero_element = self.transform(0)

    def transform(self, values):
        return (values - self.min) / (self.max - self.min + self.epsilon )

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
