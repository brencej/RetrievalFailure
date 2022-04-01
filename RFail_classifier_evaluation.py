import numpy as np
import pandas as pd
import xarray
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline

import os

output_folder = "test/"
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
    ind = [np.where(goodi)[0] for goodi in good]
    inds += [np.array(np.vstack([np.stack((np.ones(len(ind[i]))*i, ind[i]), axis=1) for i in range(len(ind))]), dtype="int")]
    print([len(ql2i) for ql2i in ql2])

os.makedirs(output_folder, exist_ok=True)
with open(output_folder + "AUC_scores.txt", "w") as f:
    f.write("product | macro AUC\n")

for n in range(len(names)):
    rad = rads[n]
    ql = qls[n]
    wl = wls[n]

    print(list(names.keys())[n])
    preds = []
    imps = np.zeros(len(wl[0][0]))

    for i in range(len(times)):
        trainX = np.vstack([rad[j] for j in range(len(ql)) if j != i])
        trainY = np.hstack([ql[j] for j in range(len(ql)) if j != i])
        testX = rad[i]
        testY = ql[i]

        model = make_pipeline(RandomUnderSampler(), 
                              StandardScaler(),
                              PCA(n_components = n_components[n]),
                              ExtraTreesClassifier(n_estimators = 200))
                             
        model.fit(trainX, trainY)
        preds += [model.predict_proba(testX)[:, 1]]
        imp = np.dot(np.abs(model[2].components_).T, model[3].feature_importances_)
        #imp = model[2].feature_importances_
        #imp = abs(model[2].coef_[0])
        imp = imp / np.max(imp) / 5
        imps += imp
    print([len(pred) for pred in preds])

    auc = roc_auc_score(np.hstack(ql), np.hstack(preds))
    with open(output_folder + "AUC_scores.txt", "a") as f:
        f.write(list(names.values())[n] + " " + str(auc) + "\n")
    
    roc = roc_curve(ql[i], preds[i])
    with open(output_folder + "ROC_" + list(names.keys())[n] + ".txt", "w") as f:
        np.savetxt(f, np.vstack(roc).T)

    with open(output_folder + "FI_" + list(names.keys())[n] + ".txt", "w") as f:
        np.savetxt(f, imps)

    indpred = pd.DataFrame()
    indpred["day"] = inds[n][:,0]
    indpred["ind"] = inds[n][:,1]
    indpred["pred"] = np.hstack(preds)
    indpred.to_csv(output_folder + "predictions_" + list(names.keys())[n] + ".csv", index=False)
    #with open(output_folder + "predictions_" + list(names.keys())[n] + ".txt", "w") as f:
    #    np.savetxt(f, np.hstack(preds))


    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(5,5))
    plt.plot(roc[0], roc[1]);
    plt.plot([0,1], [0,1], "k-")
    plt.axis("equal")
    plt.xlabel("false positive")
    plt.ylabel("true positive")
    plt.savefig(output_folder + "fig_ROC_" + list(names.keys())[n] + ".png")

    plt.figure(figsize=(8,4))
    plt.plot(wls[n][0][0], imps, "ko", ms=1)
    if fullband:
        b_ind = [np.where(wls[n][0][i] == bands[n][i])[0][0] for i in range(len(bands[n]))]
        plt.plot(bands[n], imps[b_ind], "ro", ms=1)
    plt.xlabel("wavenumber")
    plt.ylabel("ET feature importance")
    plt.tight_layout()
    plt.savefig(output_folder + "fig_ET_FI_" + list(names.keys())[n] + ".png")