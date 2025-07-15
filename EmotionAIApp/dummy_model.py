# ONLY FOR TESTING WITHOUT REAL .h5 FILE
import numpy as np

class DummyModel:
    def predict(self, data):
        # Random fake output like a real model
        result = np.zeros((1, 7))
        result[0, np.random.randint(0, 7)] = 1
        return result

# Save dummy model as pickle or just use directly in app
dummy_model = DummyModel()
