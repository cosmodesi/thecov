"""

thecov

A python package to compute analytic covariance matrices based
on Wadekar & Scoccimarro 2019 (arXiv:1910.02914).
"""

__version__ = "0.1.0"
__author__ = 'Otavio Alves'
__credits__ = 'Dark Energy Spectroscopic Instrument'

from .geometry import BoxGeometry, SurveyGeometry
from .covariance import GaussianCovariance