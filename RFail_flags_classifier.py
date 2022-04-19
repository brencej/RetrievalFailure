import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray
import os

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, classification_report, roc_auc_score


fullband = True
data_folder = "E:/datasets/ESA/data-3/"
output_folder = "results/flag_classification/"

folders = ["Global_Survey-2020-01-15", "Global_Survey-2020-03-15", "Global_Survey-2020-06-15", 
           "Global_Survey-2020-09-15", "Global_Survey-2020-12-15"]
times = ["1", "3", "6", "9", "12"]

names = {"H2O": "Products_Radiance-H2O,O3.nc", 
        "CO": "Products_Radiance-CO.nc",
        #"NH3": "Products_Radiance-NH3.nc",
        "mixed": "Products_Radiance-TATM,H2O,HDO,N2O,CH4,O3,TSUR,CLOUDEXT-bar_land.nc"}

use_PCA = True
n_components = [20]*3

flags_to_include = [
    ['Quality_Flag', 'DayNightFlag', 'DOFs',
       'LandFlag', 'KDotDL_QA', 'LDotDL_QA', 'CloudTopPressure',
       'AverageCloudEffOpticalDepth', 'CloudVariability_QA', 'O3_Ccurve_QA',
       'O3_Tropo_Consistency_QA', 'O3_ColumnErrorDU', 'RadianceResidualRMS',
       'RadianceResidualMean'],
    ['Quality_Flag', 'DayNightFlag', 'DOFs',
       'LandFlag', 'KDotDL_QA', 'LDotDL_QA', 'CloudTopPressure',
       'AverageCloudEffOpticalDepth', 'CloudVariability_QA',
       'RadianceResidualRMS', 'RadianceResidualMean'],   
    ['Quality_Flag', 'DayNightFlag', 'DOFs',
       'LandFlag', 'KDotDL_QA', 'LDotDL_QA', 'CloudTopPressure',
       'AverageCloudEffOpticalDepth', 'CloudVariability_QA',
       'RadianceResidualRMS', 'RadianceResidualMean']]


for n in range(len(names)):
    # LOAD DATA
    name = list(names.keys())[n]
    print(name)
    file = names[name]
    data = [xarray.open_dataset(data_folder + folder + "/" + folder + "/" + file) for folder in folders]
    if fullband:
        rad = [np.array(dat.RADIANCEFULLBAND) for dat in data]
        wl = [np.array(dat.FREQUENCYFULLBAND) for dat in data]
        bands = np.array(data[0].FREQUENCY)[0]

    else:
        rad = [np.array(dat.RADIANCEOBSERVED) for dat in data]
        wl = [np.array(dat.FREQUENCY) for dat in data]
    ql = [np.array(dat.QUALITY) for dat in data]
    
    # FIND ROWS WITH MISSING LABELS, REMOVE THEM FROM DATA
    #good = [qli != -999 for qli in ql]
    #ql = [1-ql[i][good[i]] for i in range(len(ql))]
    #rad = [rad[i][good[i]] for i in range(len(ql))]
    #wl = [wl[i][good[i]] for i in range(len(ql))]
    
    # STORE INDICES
    #ind = [np.where(goodi)[0] for goodi in good]
    #inds = np.array(np.vstack([np.stack((np.ones(len(ind[i]))*i, ind[i]), axis=1) for i in range(len(ind))]), dtype="int")

    # READ QUALITY FLAGS
    flags = pd.read_csv(data_folder + "/Quality_Flags_" + name + ".csv")
    #flags = flags.iloc[np.hstack(good)]

    # ARRAY OF INDICES OF MEASUREMENTS DATES, FOR STRATIFICATION 
    #day_inds = np.hstack([[i]*sum(good[i]) for i in range(len(good))])
    day_inds = np.hstack([[i]*len(rad[i]) for i in range(len(rad))])
    rad = np.vstack(rad)

    # PREPARE AUC FILE
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + name + "_AUC_scores.txt", "w") as f:
        f.write("flag | macro AUC\n")

    # MODEL
    for f, flag in enumerate(flags_to_include[n]):
        print("---", flag)
        try:
            kfold = StratifiedKFold(n_splits = 5)

            #good = flags[flag] != -999
            good = np.hstack(ql) != -999
            X = rad[good]
            y = np.array(flags[flag].astype(int))[good]

            preds = np.zeros(len(y)) 
            #imps = np.zeros(len(wl[0][0]))
            imps = []
            for train_ind, test_ind in kfold.split(X, y = day_inds[good]):
            
                if use_PCA:
                    model = make_pipeline(RandomUnderSampler(), 
                                        StandardScaler(),
                                        PCA(n_components = n_components[n]),
                                        ExtraTreesClassifier(n_estimators = 200))
                else:
                    model = make_pipeline(RandomUnderSampler(), 
                            StandardScaler(),
                            ExtraTreesClassifier(n_estimators = 200))
                                        
                model.fit(X[train_ind], y[train_ind])

                preds[test_ind] = model.predict_proba(X[test_ind])[:, 1]

                if use_PCA:
                    imp = np.dot(np.abs(model[2].components_).T, model[3].feature_importances_)
                else:
                    imp = model[2].feature_importances_

                #imp = imp / np.sum(imp) / 5
                #imps += imp
                imps += [imp]
            
            imps = np.mean(imps, axis=0)
            auc = roc_auc_score(y, np.hstack(preds))
            with open(output_folder + name + "_AUC_scores.txt", "a") as f:
                f.write(flag + " " + str(auc) + "\n")

            plt.figure(figsize=(8,4))
            plt.plot(wl[0][0], imps, "ko", ms=1)
            #if fullband:
            #    b_ind = [np.where(wl[0][i] == bands[i])[0][0] for i in range(len(bands))]
            #    plt.plot(bands, imps[b_ind], "ro", ms=1)
            plt.xlabel("wavenumber")
            plt.ylabel("ET feature importance")
            plt.title("product: " + name + ", flag: " + flag + ", AUC: " + str(round(auc, 3)))    
            #plt.ylim(-0.05,1.05)
            plt.tight_layout()
            plt.savefig(output_folder + "fig_FI_" +name + "_" + flag + ".png")
        except ValueError as e:
            print("ValueError for flag: ", flag)
            