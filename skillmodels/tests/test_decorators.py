from skillmodels.decorators import extract_params


def test_extract_params_decorator():
    @extract_params(key="a")
    def f(x, params):
        return x * params

    assert f(x=3, params={"a": 4, "b": 5}) == 12


def test_extract_params_direct_call():
    def f(x, params):
        return x * params

    g = extract_params(f, key="a")

    assert g(x=3, params={"a": 4, "b": 5}) == 12
