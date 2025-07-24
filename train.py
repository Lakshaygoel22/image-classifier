import data_loader
import model
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = data_loader.load_and_preprocess_data()
datagen = data_loader.get_data_generator()
datagen.fit(X_train)

cnn_model = model.build_model()

# Train with data augmentation
history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train)//64,
    epochs=15,
    validation_data=(X_test, y_test)
)

# Unfreeze base model and fine-tune
cnn_model.layers[0].trainable = True
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train)//64,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save model
cnn_model.save('image_classifier.h5')

# Save training history plot
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.show()
