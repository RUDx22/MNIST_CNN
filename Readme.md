Accuracy=99.7%  (70000 images)
0bb3e9e (accuracy added)

Handwritten Digit Recognition with PyTorch
This is a Convolutional Neural Network (CNN) I built using PyTorch to classify handwritten digits from the MNIST dataset. I wanted to handle the entire deep learning workflow myself, so this project covers everything from setting up the data pipeline to designing the model architecture, training, and testing.

Project Highlights
Custom Architecture: I built the network using multi-channel convolutional layers paired with max-pooling to shrink the spatial dimensions, followed by fully connected linear layers.

Preventing Overfitting: I added a Dropout layer (0.25 probability) to help the model generalize better and prevent it from just memorizing the training data.

Data Handling: I used Torchvision transforms to normalize the images (setting the mean and standard deviation to 0.5) and DataLoaders to keep the batch processing efficient.

Hardware Flexibility: The script automatically checks for a GPU (CUDA). If it finds one, it shifts the training there to speed things up; otherwise, it cleanly falls back to the CPU.

Tools Used
Python

PyTorch (torch.nn, torch.optim)

Torchvision

How the Model is Structured
The network follows a straightforward path from extracting image features to making the final classification:

Conv Layer 1: 1 input channel → 32 output channels, 3x3 kernel, padding=1 (ReLU activation + MaxPool).

Conv Layer 2: 32 input channels → 64 output channels, 3x3 kernel, padding=1 (ReLU activation + MaxPool).

Flattening: Reshaping the 2D tensor into 1D for the fully connected layers.

Fully Connected 1: 64 * 7 * 7 nodes → 128 nodes (ReLU activation).

Dropout: 25% drop probability.


Fully Connected 2 (Output): 128 nodes → 10 nodes (representing the digits 0-9).

Fully Connected 2 (Output): 128 nodes → 10 nodes (representing the digits 0-9).
