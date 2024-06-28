# Import libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Task 1: Setup and Data Verification
# Path to the dataset
dataset_path = r"C:\\Users\\Francesco Corda\\Documents\\GitHub\\Project_CV\\images\\images"

# Verify the dataset path
if not os.path.exists(dataset_path):
    print(f"The dataset path {dataset_path} does not exist.")
else:
    # List categories
    categories = os.listdir(dataset_path)
    print(f"Categories found: {categories}")

    # Count the number of images in each category
    category_counts = {}
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        default_path = os.path.join(category_path, 'default')
        real_world_path = os.path.join(category_path, 'real_world')

        # Verify the existence of default and real_world directories
        if not os.path.exists(default_path):
            print(f"The default folder does not exist for category {category}")
            continue
        if not os.path.exists(real_world_path):
            print(f"The real_world folder does not exist for category {category}")
            continue

        default_images = os.listdir(default_path)
        real_world_images = os.listdir(real_world_path)
        category_counts[category] = len(default_images) + len(real_world_images)

# Task 2: Data Visualization
    # Plot the distribution of images across categories
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.xticks(rotation=90)
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Categories')
    plt.show()

    # Function to visualize sample images
    def show_sample_images(category, image_type='default'):
        category_path = os.path.join(dataset_path, category, image_type)
        sample_images = os.listdir(category_path)[:5]  # Show first 5 images
        plt.figure(figsize=(15, 5))
        for i, image_name in enumerate(sample_images):
            image_path = os.path.join(category_path, image_name)
            img = Image.open(image_path)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(f'{category} - {image_type}')
            plt.axis('off')
        plt.show()

    # Show sample images from a few categories
    for category in categories[:3]:  # Adjust the range to select different categories
        show_sample_images(category, 'default')
        show_sample_images(category, 'real_world')

    # Image dimensions and batch size
    IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduce the image size to help decreasing the computational load.
    BATCH_SIZE = 16  # Smaller batch size. This determine the number of samples that will be propagated through the network at once

    # Simplified ImageDataGenerator with minimal augmentation. This is a class in Keras that generates batches of tensor image data with real-time data augmentation.
    # Explanation: This configuration applies minimal augmentation, including rescaling the pixel values to the range [0, 1], splitting the data for validation, 
    # and applying horizontal flipping and zooming for augmentation. This helps the model generalize better by introducing slight variations in the training data. 
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,  # 20% of data for validation
        horizontal_flip=True,
        zoom_range=0.2
    )

    # Training and validation data generators
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True  # Ensure shuffling
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # No need to shuffle for validation
    )

# Task 4: Model Definition and Training
    # Load the MobileNetV2 model with pre-trained weights, exclude top layers. 

    # Explanation: MobileNetV2 is a lightweight, efficient convolutional neural network model designed for mobile and embedded vision applications. 
    # It is designed to be both fast and accurate, making it suitable for real-time applications on devices with limited computational power. 
    # It uses depthwise separable convolutions to reduce the number of parameters and computational cost.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Unfreeze the last few layers of the base model for fine-tuning
    # Explanation: allows these layers to be trainable, meaning their weights can be updated during training. 
    # This is done to fine-tune the model on the new dataset, helping the model to better adapt to the specific features of the new data while retaining the learned features from the pre-trained model.
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Build your model on top of it. The 'Sequential' is a linear stack of layers in Keras. It is a convenient way to build a neural network layer-by-layer
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Adjust complexity
        Dropout(0.3),  # Regularization
        Dense(len(categories), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Compute class weights. It helps to handle class imbalance in the dataset.
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    # Early stopping callback. It helps to prevent overfitting by monitoring the validation loss during training and stopping the training process when the validation loss stops improving for a specified number of epochs (patience)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Calculate steps_per_epoch and validation_steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE #number of batches of samples to be processed before declaring one epoch complete.
    validation_steps = validation_generator.samples // BATCH_SIZE # number of batches of samples to be processed before declaring one validation round complete.

    # Train the model with class weights
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,  # Reduce the number of epochs
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    # Evaluate the model using appropriate metrics and visualize the results
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # Get the ground truth labels
    y_true = validation_generator.classes

# Task 5: Model Evaluation
    # Classification report
    class_labels = list(validation_generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()
