__version__ = "0.0.6"

from .explainers._kernel import KernelExplainer
from .explainers._linear import LinearExplainer
from .plots._beeswarm import beeswarm as beeswarm_plot
from .plots._force import force as force_plot
from .plots._force import initjs, save_html
from .plots._summary import summary_legacy as summary_plot
from .utils._legacy import kmeans

__all__ = [
    "KernelExplainer",
    "LinearExplainer",
    "kmeans",
    "summary_plot",
    "beeswarm_plot",
    "force_plot",
    "initjs",
    "save_html",
]
