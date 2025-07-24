from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import data_loader

model = load_model('image_classifier.h5')
_, _, X_test, y_test = data_loader.load_and_preprocess_data()

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")
