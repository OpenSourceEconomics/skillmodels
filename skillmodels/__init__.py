__version__ = "0.2.2"


msg = (
    "Skillmodels requires jax>=0.2.0 to be installed. It is not installed "
    "automatically from conda-foreg because that version does not support GPUS. "
    "If you want to run skillmodels on CPUs (i.e. on a normal computer like a laptop) "
    "you can simply install jax via 'conda install -c conda-forge jax>=0.2.0'. "
    "If you need GPU support, follow this guide: "
    "https://jax.readthedocs.io/en/latest/developer.html#"
)

try:
    from jax import config  # noqa
except ImportError:
    raise ImportError(msg)
config.update("jax_enable_x64", True)
