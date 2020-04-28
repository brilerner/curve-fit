import holoviews as hv
from holoviews import dim, opts
hv.extension('plotly')
import numpy as np
import panel as pn
pn.extension()
import param

import string
from itertools import cycle
from collections import OrderedDict
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import time

import funcs

funcs.Fit().view()
