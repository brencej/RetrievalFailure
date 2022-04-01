# Predictions of satellite retrieval failures of air quality using machine learning

## Requirements
- python 3.7 or newer
- numpy
- pandas
- xarray
- matplotlib
- joblib
- scikit-learn (sklearn)
- imbalanced-learn (imblearn)


## Scripts
### **RFail_classifier_evaluation.py**
Constructs and evaluates a pipeline for each air quality product. The data consists of measurements of five different days in the year. To evaluate, we train on four of the days and predict the fifth. We repeat this for each choice of the fifth day and pool the predictions. 

The output of the script are ROC curves and AUC scores.

Settings can currently only be changed in the script itself. They include:
- output_folder (string): subfolder that will be created and will store results
- data_folder (string): the path to the folder with the data (not included in repository)
- fullband (bool): True - train on entire measured spectrum, False - train on selected wavebands
- n_components (list of integers): number of principal components to use. Each element of the list corresponds to one of the air quality product, in the same order as given in *names*. Default order: H20, CO, mixed (TATM).

### **RFail_classifier_train.py**
Constructs a pipeline and trains the model for each air quality product on the entire given data. The model is exported using *joblib*.

Uses the same settings as **RFail_classifier_evaluation.py**.

## Models
The exported models for CO and TATM are too large for github. They can be trained using **RFail_classifier_train.py** or downloaded from this [Dropbox link](https://www.dropbox.com/sh/ypd8ynien8qc5ed/AAB74LNBWLtAQRGMxFEx5U_Ra?dl=0).

The loading and using of trained models is demonstrated in **example.py**

Loading and using pre-trained models has the same requirements as training them.