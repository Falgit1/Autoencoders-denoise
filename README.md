## Step 1: Import Libraries  
```python
import numpy as np  
import tensorflow as tf  
from tensorflow.keras import layers, models  
import matplotlib.pyplot as plt  
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
```

## Explanation:  
- Loads MNIST dataset and necessary libraries.  
- Only image data is used; labels are not required for autoencoders.

---

## Step 2: Normalize and Reshape Data  
```python
x_train = x_train.astype('float32') / 255.0  
x_test = x_test.astype('float32') / 255.0  

x_train = np.expand_dims(x_train, axis=-1)  
x_test = np.expand_dims(x_test, axis=-1)
```

## Explanation:  
- Pixel values scaled to [0, 1] range.  
- Shape expanded to include a channel dimension: `(28, 28, 1)`.

---

## Step 3: Visualize Original Image  
```python
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')  
plt.show()
```

## Explanation:  
- Displays a sample clean image from the dataset.

---

## Step 4: Add Noise to the Images  
```python
def add_noise(imgs, noise_factor=0.5):  
    noisy_imgs = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)  
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)  
    return noisy_imgs  

x_train_noisy = add_noise(x_train)  
x_test_noisy = add_noise(x_test)
```

## Explanation:  
- Adds Gaussian noise to the input images.  
- Result is clipped to stay in the valid pixel range.

---

## Step 5: Visualize Noisy Image  
```python
plt.imshow(x_train_noisy[0].reshape(28, 28), cmap='gray')  
plt.show()
```

## Explanation:  
- Shows a noisy version of a training image.

---

## Step 6: Build Denoising Autoencoder  
```python
def build_autoencoder():  
    input_img = layers.Input(shape=(28, 28, 1))  
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)  
    x = layers.UpSampling2D((2, 2))(x)  
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  
    x = layers.UpSampling2D((2, 2))(x)  
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  
    
    autoencoder = models.Model(input_img, decoded)  
    return autoencoder
```

## Explanation:  
- **Encoder**: Compresses input using Conv2D + MaxPooling.  
- **Decoder**: Reconstructs the input using Conv2D + UpSampling.  
- **Output**: A denoised image.

---

## Step 7: Train the Autoencoder  
```python
autoencoder = build_autoencoder()  
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')  
autoencoder.fit(x_train_noisy, x_train, epochs=50, batch_size=128, validation_data=(x_test_noisy, x_test))
```

## Explanation:  
- Trains the model to predict clean images from noisy inputs.  
- Uses binary crossentropy loss.

---

## Step 8: Predict and Display Denoised Output  
```python
denoised_images = autoencoder.predict(x_test_noisy)

plt.figure(figsize=(20, 4))  
n = 10  # number of images to display  
for i in range(n):  
    ax = plt.subplot(2, n, i + 1)  
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')  
    ax.get_xaxis().set_visible(False)  
    ax.get_yaxis().set_visible(False)  
    
    ax = plt.subplot(2, n, i + 1 + n)  
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')  
    ax.get_xaxis().set_visible(False)  
    ax.get_yaxis().set_visible(False)  
plt.show()
```

## Explanation:  
- Compares noisy input (top row) to denoised output (bottom row).

---

## Step 9: Save the Model *(Optional)*  
```python
# autoencoder.save('autoencoder_denoise_model.h5')
```

## Explanation:  
- Saves the trained model to reuse for denoising tasks.

---

## Step 10: Visualize the Effects of Different Noise Settings  
```python
blank_img = np.zeros((28, 28))  
noise_factor = 1.0  
loc_values = [0.0, 0.5, 1.0]  
scale_values = [0.5, 1.0, 2.0]  

for loc in loc_values:  
    for scale in scale_values:  
        noisy_img = blank_img + noise_factor * np.random.normal(loc=loc, scale=scale, size=blank_img.shape)  
        noisy_img = np.clip(noisy_img, 0, 1)  

        plt.figure()  
        plt.title(f"loc={loc}, scale={scale}")  
        plt.imshow(noisy_img, cmap='gray')  
        plt.colorbar()  
        plt.show()
```

## Explanation:  
- Shows how different noise parameters affect blank images.  
- Helps visualize the shape/distribution of Gaussian noise.

---

## Step 11: Test on a Real Image  
```python
import cv2  
from tensorflow.keras.models import load_model  

autoencoder = load_model("autoencoder_denoise_model.h5")  
img = cv2.imread("img1.png")  
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
grayscale_img = cv2.resize(grayscale_img, (28, 28))  
grayscale_array = grayscale_img.astype("float32") / 255.0  
img = np.expand_dims(grayscale_array, axis=-1)

plt.imshow(img, cmap="gray")  
plt.show()

img = np.expand_dims(img, axis=0)  
img2 = img + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=img.shape)  
img2 = np.clip(img2, 0., 1.)  
predicted_img = autoencoder.predict(img2)

imgs = [img, img2, predicted_img]  
for i in range(1, 4):  
    plt.subplot(1, 3, i)  
    plt.imshow(imgs[i-1][0], cmap="gray")  
plt.show()
```

## Explanation:  
- Loads an external image, preprocesses it, and adds noise.  
- Uses the trained autoencoder to denoise the image and visualize results.



