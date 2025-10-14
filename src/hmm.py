import numpy as np
from hmmlearn import hmm

# Define the model — GaussianHMM for continuous data
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)

# Create synthetic sequential data (e.g., financial transaction patterns)
X = np.array([
    [0.2], [0.3], [0.5], [1.2], [1.3], [1.5], [2.0], [2.2], [2.3], [3.0]
])
lengths = [len(X)]  # Single sequence

# Fit the model
model.fit(X, lengths)

# Predict hidden states for each observation
hidden_states = model.predict(X)

# Predict the next likely observation
next_obs, _ = model.sample(1)

print("Observed sequence:\n", X.flatten())
print("\nPredicted hidden states:\n", hidden_states)
print("\nPredicted next likely observation value:", next_obs.flatten()[0])
