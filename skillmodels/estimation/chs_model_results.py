from statsmodels.base.model import GenericLikelihoodModelResults
from statsmodels.tools.decorators import resettable_cache, cache_readonly


class CHSModelResults(GenericLikelihoodModelResults):
    def __init__(self, model, mlefit, optimize_dict):
        self.model = model
        self.optimize_dict = optimize_dict
        self.nobs = model.nobs
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = resettable_cache()
        self.__dict__.update(mlefit.__dict__)
        self.param_names = model.param_names(params_type='long')

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        raise NotImplementedError(
            'The summary method is not yet implemented for CHSModelResults')

    @cache_readonly
    def df_modelwc(self):
        return self.df_model

    @cache_readonly
    def llf(self):
        return self.optimize_dict['log_lh_value']

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        raise NotImplementedError(
            'Bootstrap is not yet implemented for CHSModelResults')

    def get_nlfun(self, fun):
        raise NotImplementedError

    def save(self, fname, remove_data=False):
        raise NotImplementedError(
            'A save method is not yet implemented for CHSModelResults')

    @classmethod
    def load(cls, fname):
        raise NotImplementedError(
            'A load method is not yet implemented for CHSModelResults')

    def remove_data(self):
        raise NotImplementedError(
            'A remove_data method is not yet implemented for CHSModelResults')

    def predict(self, exog=None, transform=True, *args, **kwargs):
        raise NotImplementedError(
            'A predict method is not yet implemented for CHSModelResults')

    @cache_readonly
    def bic(self):
        raise NotImplementedError

    @cache_readonly
    def score_obsv(self):
        raise NotImplementedError

    @cache_readonly
    def hessv(self):
        raise NotImplementedError

    @cache_readonly
    def covjac(self):
        raise NotImplementedError

    @cache_readonly
    def covjhj(self):
        raise NotImplementedError

    @cache_readonly
    def bsejhj(self):
        raise NotImplementedError

    @cache_readonly
    def bsejac(self):
        raise NotImplementedError



