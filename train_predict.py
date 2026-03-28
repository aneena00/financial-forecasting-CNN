import numpy as np
from sklearn.model_selection import train_test_split
from model_cnn import build_cnn_model

def run_experiment(X, y):
    # CNN expects 4D input: (batch, height, width, channels)
    X = X[..., np.newaxis] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = build_cnn_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    
    predictions = model.predict(X_test)
    return y_test, predictions
