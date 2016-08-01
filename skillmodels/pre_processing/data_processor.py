from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor
import numpy as np
# todo make nicer import of the dataset


class DataProcessor:
    """Transform a pandas DataFrame in long format into numpy arrays."""

    def __init__(self, model_name, dataset_name, model_dict, dataset):
        specs = ModelSpecProcessor(
            model_name, dataset_name, model_dict, dataset)
        self.data = specs._data
        self.add_constant = specs.add_constant
        self.periods = specs.periods
        self.controls = specs.controls
        self.obs_to_keep = specs.obs_to_keep
        self.nupdates = specs.nupdates
        self.nobs = specs.nobs
        self.update_info = specs.update_info()

    def c_data(self):
        """A List of 2d arrays with control variables for each period.

        The arrays are of the shape[nind, number of control variables in t].

        """
        c_data = []
        self.data['constant'] = 1.0
        const_list = ['constant'] if self.add_constant is True else []

        for t in self.periods:
            df = self.data[self.data['period'] == t]
            arr = df[const_list + self.controls[t]].values[self.obs_to_keep]
            c_data.append(arr)
        return c_data

    def y_data(self):
        """A 2d numpy array that holds the measurement variables.

        The array is of shape [nupdates, nind].

        """
        dims = (self.nupdates, self.nobs)
        y_data = np.zeros(dims)

        counter = 0
        for t in self.periods:
            measurements = list(self.update_info.loc[t].index)
            df = self.data[self.data['period'] == t][measurements]

            y_data[counter: counter + len(measurements), :] = \
                df.values[self.obs_to_keep].T
            counter += len(measurements)
        return y_data
