import numpy as np
import pandas as pd
import os
import xarray
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


output_folder = "models/"
fullband = True

data_folder = "E:/datasets/ESA/data-3/"

n_components = [20]*3

folders = ["Global_Survey-2020-01-15", "Global_Survey-2020-03-15", "Global_Survey-2020-06-15", 
           "Global_Survey-2020-09-15", "Global_Survey-2020-12-15"]
times = ["1", "3", "6", "9", "12"]

names = {"H2O": "Products_Radiance-H2O,O3.nc", 
        "CO": "Products_Radiance-CO.nc",
        #"NH3": "Products_Radiance-NH3.nc",
        "mixed": "Products_Radiance-TATM,H2O,HDO,N2O,CH4,O3,TSUR,CLOUDEXT-bar_land.nc"}


wls = []
qls = []
rads = []
bands = []
inds = []

for name in names:
    file = names[name]
    data = [xarray.open_dataset(data_folder + folder + "/" + folder + "/" + file) for folder in folders]
    if fullband:
        rad = [np.array(dat.RADIANCEFULLBAND) for dat in data]
        wl = [np.array(dat.FREQUENCYFULLBAND) for dat in data]
        bands += [np.array(data[0].FREQUENCY)[0]]

    else:
        rad = [np.array(dat.RADIANCEOBSERVED) for dat in data]
        wl = [np.array(dat.FREQUENCY) for dat in data]
    ql = [np.array(dat.QUALITY) for dat in data]

    good = [qli != -999 for qli in ql]
    ql2 = [1-ql[i][good[i]] for i in range(len(ql))]
    rad2 = [rad[i][good[i]] for i in range(len(ql))]
    wl2 = [wl[i][good[i]] for i in range(len(ql))]
    wls += [wl2]
    rads += [rad2]
    qls += [ql2]

os.makedirs(output_folder, exist_ok=True)

for n in range(len(names)):
    rad = rads[n]
    ql = qls[n]
    wl = wls[n]
    name = list(names.keys())[n]

    print(name)

    X = np.vstack(rad)
    Y = np.hstack(ql)

    model = make_pipeline(RandomUnderSampler(), 
                            StandardScaler(),
                            PCA(n_components = n_components[n]),
                            ExtraTreesClassifier(n_estimators = 200))
                            
    model.fit(X, Y)
    
    joblib.dump(model, output_folder + name + "_model.joblib", compress = 3)