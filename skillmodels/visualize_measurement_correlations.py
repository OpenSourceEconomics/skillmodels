"""Visualize the correlations of measurement variables in a given period."""
import mathplotlib.pyplot as plt
import seaborn as sns

from skillmodels.process_model import process_model


def visualize_measurement_correlations(
    period,
    model_dict,
    data,
    factors=None,
    sns_kwargs=None,
    decimals=2,
    show_unit_corr=False,
):
    """Plot correlation heatmaps for factor measurements.
    Args:
        period(int): the period in which to calculate the correlations.
        model_dict(dct): dictionary of model attributes to be passed to process_model
            and extract measurements for each period.
        factors(list): list of factors, whose measurement correlation to calculate. If
            the default value of None is passed, then calculate and plot correlations of
            all measurements.
        sns_kwargs(dct): dictionary of key word arguments to pass to sns.heatmap().
            If the default value of None is passed, the default kwargs defined in the
            function will be used.
        decimals(int): number of digits after the decimal point to round the correlation
            values to. The default value is 2.
        unit_corr(bool): a boolean variable that displays unit correlations on the
            heatmap is set to True. The default value is False.
    Returns:
        fig(pyplot figure): the figure with correlaiton heatmap.

    """
    model = process_model(model_dict)
    period_name = model["update_info"].index.names[0]
    period_info = model["update_info"].loc[period].reset_index()
    measurements = []
    if factors:
        for fac in factors:
            measurements += period_info.query(
                f"{fac} == True and purpose == 'measurement'"
            )["variable"].to_list()
    else:
        measurements = period_info.query(f"purpose == 'measurement'")[
            "variable"
        ].to_list()
    df = data.reset_index().query(f"{period_name}=={period}")[measurements]
    corr = df.corr().round(decimals)
    if not show_unit_corr:
        labels = corr.replace(1, "")
    else:
        labels = corr
    fig, ax = plt.subplots(
        1, 1, figsize=(6 + 1.05 * len(measurements), 3 + len(measurements))
    )
    kwargs = {
        "cmap": "coolwarm",
        "center": 0,
        "annot": labels,
        "fmt": "",
        "annot_kws": {"fontsize": 12},
    }
    if sns_kwargs:
        kwargs.update(sns_kwargs)
    sns.heatmap(corr, ax=ax, **kwargs)
    plt.title(f"{period_name}: {period}")
    fig.tight_layout()
    plt.close()
    return fig
