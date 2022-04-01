import numpy as np
import joblib

# load a model pipeline
model = joblib.load("models/CO_model.joblib")

# lets create a fake spectrum
X = np.random.normal(loc=0, scale=1, size=2223)
# the model expects an array of samples, not just one row,
# so we need to reshape it
X = X.reshape((1, -1))

# Make a prediction using the model.
# The predictions are probabilities for each class,
# we take only the probability of the positive class,
# since they sum up to 1 anyway.
pred_prob = model.predict_proba(X)[:, 1]

# the predictions are a list of probabilities, one for each sample
print("We predict a", pred_prob[0]*100, "% probability that retrieval will fail.")

# to predict classes, we choose a probability threshold
def probability_to_class(predicted_probabilities, threshold = 0.5):
    return predicted_probabilities >= threshold

print("Predicted class with a 35% threshold:", probability_to_class(pred_prob, 0.35)[0])