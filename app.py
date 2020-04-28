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


# Gaussian functions

def ngauss(*args):
    x, par = args[0], args[1:]
    groups = [par[int(i*3):int(3+i*3)] for i in range(int(len(par)/3))]
    curves = [gauss(x, amp, cen, sig) for amp, cen, sig in groups if sig>0]
    return np.sum(curves,axis=0)

def gauss(_x_array, amp, cen, sig):
    return amp*(1/(sig*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((_x_array-cen)/sig)**2)))


def sample_gauss(n_pks):
    def make_gauss(_x_array):
        amp = np.random.uniform(10,90)
        cen = np.random.uniform(20,80)
        sig = np.random.uniform(1,3)
        return gauss(_x_array, amp, cen, sig)
    
    x_array = np.linspace(0,101,1000)
    y_array = sum([make_gauss(x_array) for i in range(n_pks)])
    # creating some noise to add the the y-axis data
    y_noise = (np.exp((np.random.ranf(len(x_array)))))/2
    y_array += y_noise
    y_array = y_array - np.average(y_noise) # bring baseline to 0
    return np.column_stack((x_array, y_array))


# Fitting functions

def guesser(data):
    x,y = data.T
    yfilt = savgol_filter(y,101,5)
    peaks, props = find_peaks(yfilt, height=0.1*y.max(), width=(None,None))
    amps = y[peaks] # convert to gaussian amps below
    cens = x[peaks]
    sigs = x[list(map(int, props['right_ips']))] - x[list(map(int, props['left_ips']))]
    tups = [(amp*(sig*(np.sqrt(2*np.pi))), cen, sig) for amp,cen,sig in zip(amps,cens,sigs)]
    return tups


def optimizer(data, p0):
    x,y = data.T
    popt_gauss, pcov_gauss = curve_fit(ngauss, x, y, p0=p0)
    tups = [popt_gauss[i*3:3+3*i] for i in range(len(popt_gauss)//3)]
    return tups



# Classes

class Buttons():

    def __init__(self, obj):
        # style opts
        self.back = pn.widgets.Button(name='<<<', width=200, button_type='primary')
        self.forward = pn.widgets.Button(name='>>>', width=200, button_type='primary')
        self.add = pn.widgets.Button(name='Add Fit', width=120, button_type='primary')
        self.remove = pn.widgets.Button(name='X', width=80, button_type='danger')
        self.reset = pn.widgets.Button(name='RST', width=80, button_type='warning')
        self.optimize = pn.widgets.Button(name='Optimize', button_type='primary', width=75, height=50)
        self.guess = pn.widgets.Button(name='Guess', button_type='primary', width=75, height=50)
        self.restart = pn.widgets.Button(name='RESTART', button_type='warning', width=75, height=75)
        
        self.obj = obj
        
        ## KEEP IN INIT ##
        
        # functionality
        def shift(incr):
            trials = self.obj.param.trial.objects[::incr]
            trial = self.obj.trial
            index = trials.index(trial)
            self.obj.trial = next(cycle((trials[index:] + trials[:index])[1:]))
        def _back(event):
            shift(-1)
        def _forward(event):
            shift(1)
        def _add(event):
            self.obj.add_fit()
        def _remove(event):
            self.obj.remove_fit()
        def _reset(event):
            self.obj.reset_fit()
        def _guess(event):
            self.obj.guess_fit()
        def _optimize(event):
            self.obj.optimize_fit()
        def _restart(event):
            self.obj.restart_class()
            
        # on click behavior
        self.back.on_click(_back)
        self.forward.on_click(_forward)
        self.add.on_click(_add)
        self.remove.on_click(_remove)
        self.reset.on_click(_reset) 
        self.guess.on_click(_guess) 
        self.optimize.on_click(_optimize) 
        self.restart.on_click(_restart) 
        

class Fit_Params(param.Parameterized):
    step = 0.0001
    amp = param.Number(50, bounds=(0,600), step=step, doc="Amp")
    cen = param.Number(50, bounds=(0,100), step=step, doc="Cen")
    sig = param.Number(5, bounds=(0,10), inclusive_bounds=(False,True), step=step, doc="Sig")
    obj = param.ObjectSelector(precedence=-1) # for instance from Curve class
    fit_type = param.String('Gaussian')
    
    def __init__(self, **params):
        super(Fit_Params, self).__init__(**params)
        self.buttons = Buttons(self)

    def view(self):
        slider_wid = 100
        amp_widg =  pn.Param( self.param.amp, width=slider_wid, widgets={'amp':{'type':pn.widgets.FloatSlider}} )
        cen_widg =  pn.Param( self.param.cen, width=slider_wid, widgets={'cen':{'type':pn.widgets.FloatSlider}} )
        sig_widg =  pn.Param( self.param.sig, width=slider_wid, widgets={'sig':{'type':pn.widgets.FloatSlider}} )

        modifiers = pn.Column( self.buttons.remove, self.buttons.reset, align='center')
        fit_widgets = pn.Column(amp_widg, cen_widg, sig_widg)
        return pn.Column(self.fit_type, fit_widgets, modifiers)
    
    @param.depends('amp','cen','sig', watch=True)
    def update(self):
        self.obj.updating += 1
       
    def remove_fit(self):
        self.obj.fit_param_groups.remove(self)
        self.obj.updating += 1
        
    def reset_fit(self):
        self.amp = 50
        self.cen = 50
        self.sig = 50
        self.obj.updating += 1

class Curve(param.Parameterized):

    updating = param.Number(0, precedence=-1)
    removing = param.Number(-1, precedence=-1)
    peaks = param.Integer(default=1, bounds=(1,10), precedence=1)
    
    def __init__(self, data, label, **params):
        super(Curve, self).__init__(**params)
        self.data = data
        self.label = label
        self.fit_param_groups = [Fit_Params(obj=self)]
        self.fit_buttons = Buttons(self)
        
    def fit_cols(self):
        cols = [i.view() for i in self.fit_param_groups] + [pn.Column(self.fit_buttons.add)]
        return pn.Row(objects=cols)

    def add_fit(self):
        self.fit_param_groups.append(Fit_Params(obj=self, amp=20, cen=50, sig=5))
        self.updating += 1
        
    def guess_fit(self):
        guessed_fit_tups = guesser(self.data)
        self.fit_param_groups = [Fit_Params(obj=self, amp=amp, cen=cen, sig=sig) for amp,cen,sig in guessed_fit_tups]
        self.updating += 1
        
    def optimize_fit(self):
        try:
            optimized_fit_tups = optimizer(self.data, self.get_fit_param_list())
            self.fit_param_groups = [Fit_Params(obj=self, amp=amp, cen=cen, sig=sig) for amp,cen,sig in optimized_fit_tups]
            self.updating += 1
            self.optimize_watcher = True
        except ValueError:
            pass
        
    def get_fit_param_list(self):
        groups = [[i.amp, i.cen, i.sig] for i in self.fit_param_groups]
        return [j for i in groups for j in i]


class Fit(param.Parameterized):
    
    # select number of curves and max pks/curve for sample data
    n_curves = 4 
    max_n_pks = 2 
    poss_pks = [int(np.random.choice(np.arange(1,_max_n_pks+1))) for i,_max_n_pks in zip(range(n_curves),[max_n_pks]*n_curves)]
    # generate sample data
    labels = list(string.ascii_uppercase)[:n_curves]
    data = [sample_gauss(i) for i in poss_pks]
    # put data in object selector
    curve_dict = {i.label:i for i in [Curve(data=i,label=j, peaks=k) for i,j,k in zip(data,labels,poss_pks)]}
    trial = param.Selector(objects=curve_dict)
    restart_watcher = param.Number(0)

    def __init__(self, **params):
        super(Fit, self).__init__(**params)
        self.buttons = Buttons(self)
        
    def restart_class(self):
        for i in self.param.trial.objects:
            i.data = sample_gauss(np.random.choice(self.poss_pks)) 
            i.fit_param_groups = [Fit_Params(obj=i)]
        self.restart_watcher += 1
    
    def view(self):
        _trial =  pn.Param( self.param.trial, width=150, widgets={'trial':{'type':pn.widgets.Select}} )
        trial_panel = pn.Column(_trial, self.buttons.restart)
        scroll = pn.Row(self.buttons.back, self.buttons.forward, align='center')        
        plot_panel = pn.Column(self.plot, scroll)
        fitters = pn.Column(self._guess, self._optimize, align='center')
        fit_panel = pn.Row(fitters, self.dialogs)
        
        return pn.Row(trial_panel, plot_panel, fit_panel, background='#d3d6d0')
    
    @param.depends('trial.peaks', watch=True)
    def change_pks(self):
        self.trial.data = sample_gauss(self.trial.peaks)
        self.trial.updating += 1

    @param.depends('trial', 'trial.updating', 'restart_watcher', watch=True)
    def dialogs(self):
        cols = [i.view() for i in self.trial.fit_param_groups]
        return pn.Row(pn.Row(objects=cols, background='#f0f0f0', margin=(25,5)), pn.Column(self.trial.fit_buttons.add, align='center'))
     
    def _guess(self):
        return pn.Row(Buttons(self.trial).guess)

    def _optimize(self):
        return pn.Row(Buttons(self.trial).optimize)
    
    def _restart(self):
        return pn.Row(self.buttons.restart)
        
    @param.depends('trial', 'trial.updating', 'restart_watcher', watch=True)
    def plot(self):
        
        plot = trial_curve = hv.Curve(self.trial.data)
        _fit_params = self.trial.get_fit_param_list()

        if _fit_params:
            x,y = self.trial.data.T
            fitted_ydata = ngauss(x, *_fit_params)
            fitted_data = np.column_stack((x, fitted_ydata))
            fitted_opts = opts.Curve(color='black', interpolation='linear', dash='dash')
            fitted_curve = hv.Curve(fitted_data).opts(fitted_opts)
            plot *= fitted_curve
        max_y = max([max(i.data.T[1]) for i in self.param.trial.objects])
        plot_opts = opts.Curve(width=500, height=300, xticks=5, ylim=(0,max_y))
        plot.opts(plot_opts)
        
        return plot

    def shift(self, incr):
        trials = self.param.trial.objects[::incr]
        index = trials.index(self.trial)
        self.trial = next(cycle((trials[index:] + trials[:index])[1:]))

a = Fit()
a.view()
