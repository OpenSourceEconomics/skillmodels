import numpy as np
# todo make nicer import of the dataset


class DataProcessor:
    """Transform a pandas DataFrame in long format into numpy arrays."""

    def __init__(self, specs_processor_attribute_dict):
        self.__dict__.update(specs_processor_attribute_dict)

    def c_data_chs(self):
        """A List of 2d arrays with control variables for each period.

        The arrays are of the shape[nind, number of control variables in t].

        """
        c_data = []
        self.data['constant'] = 1.0
        const_list = ['constant']

        for t in self.periods:
            df = self.data[self.data[self.period_identifier] == t]
            arr = df[const_list + self.controls[t]].values[self.obs_to_keep]
            c_data.append(arr)
        return c_data

    def y_data_chs(self):
        """A 2d numpy array that holds the measurement variables.

        The array is of shape [nupdates, nind].

        """
        dims = (self.nupdates, self.nobs)
        y_data = np.zeros(dims)

        counter = 0
        for t in self.periods:
            measurements = list(self.update_info.loc[t].index)
            df = self.data[self.data[self.period_identifier] == t][
                measurements]

            y_data[counter: counter + len(measurements), :] = \
                df.values[self.obs_to_keep].T
            counter += len(measurements)
        return y_data

    def y_data_wa(self):
        """List of DataFrames with measurement variables for each period."""
        df_list = []
        for t in self.periods:
            measurements = list(self.update_info.loc[t].index)
            df = self.data[self.data[self.period_identifier] == t][
                measurements]

            if t > 0:
                for f, factor in enumerate(self.factors):
                    if self.transition_names[f] == 'constant':
                        initial_meas = self.measurements[factor][0]
                        df[['{}_copied'.format(m) for m in initial_meas]] = \
                            df_list[0][initial_meas]

            df_list.append(df)
        return df_list

    def y_data(self):
        if self.estimator == 'chs':
            return self.y_data_chs()
        elif self.estimator == 'wa':
            return self.y_data_wa()
        else:
            raise NotImplementedError(
                'DataProcessor.y_data only works for CHS and WA estimator.')

    def c_data(self):
        if self.estimator == 'chs':
            return self.c_data_chs()
        else:
            raise NotImplementedError(
                'DataProcessor.c_data only works for CHS estimator')
