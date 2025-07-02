import os
from pathlib import Path
import sys

#sys.path.append('../utils')
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.weather_helper import load_ds_inputs
from utils.weather_helper import plot_weather

if __name__=="__main__":
    # load data
    PROJ_PATH = Path.cwd().parent
    ds_true_in = load_ds_inputs(PROJ_PATH)

    # plot data
    plot_weather(ds_true_in)