
# ASL Character Classification Using VGG16 - Jupyter Notebook Guide

This project entails building and training a machine learning model using the VGG16 architecture for classifying American Sign Language (ASL) characters. The model is trained on a dataset of images representing ASL characters. The implementation is provided in a Jupyter Notebook.

## Dependencies

To run this project, the following libraries and packages need to be installed. The listed versions are recommended to ensure compatibility, but the project may also work with other versions.

### Required Libraries and Packages:
- **Python**: Version 3.6 or higher.
- **TensorFlow**: Version 2.x. It is the core machine learning library used for building and training the neural network.
- **NumPy**: Used for numerical computations and handling arrays.
- **scikit-learn**: Required for splitting the dataset and evaluating the model.

### Installation Instructions:
Make sure Python is installed on your system. If not, you can download and install it from [Python's official website](https://www.python.org/downloads/).

Once Python is installed, you can install the required packages using pip, Python's package installer. Run the following commands in your terminal or command prompt:

```bash
pip install tensorflow==2.6.0 numpy==1.19.5 scikit-learn==0.24.2
```

Note: If you are using a GPU for training, make sure to install the GPU version of TensorFlow by replacing `tensorflow==2.6.0` with `tensorflow-gpu==2.6.0` in the above command. Also, ensure that you have the necessary CUDA and cuDNN libraries installed for GPU support.

### Verifying Installation:
After installation, you can verify the successful installation of these packages by running the following commands in your Python environment:

```python
import tensorflow as tf
import numpy as np
import sklearn

print("TensorFlow Version:", tf.__version__)
print("NumPy Version:", np.__version__)
print("scikit-learn Version:", sklearn.__version__)
```

This will output the versions of the installed packages, confirming their presence in your environment.


## Running the Notebook

1. Launch Jupyter Notebook in your environment. This can typically be done by running `jupyter notebook` in your command line or terminal.

2. Navigate to the location of the downloaded notebook file in the Jupyter Notebook interface.

3. Open the notebook to interact with it.

## Notebook Structure

The notebook is organized into multiple cells, each handling specific tasks:

1. **Import Statements**: Cells at the beginning of the notebook import all necessary libraries.

2. **Function Definitions**: The following functions are defined for various tasks:
   - `load_and_preprocess_data`: Loads the image data and labels from `.npy` files and preprocesses them.
   - `image_batch_generator`: Creates a generator that yields image data in batches.
   - `build_model`: Constructs the VGG16-based neural network model.
   - `train_model`: Trains the model using generated batches of data.
   - `save_model`: Saves the trained model to a file.
   - `evaluate_model_accuracy`: Calculates the model's accuracy on the test set.

3. **Model Training**: Cells in this section will execute the functions to load data, preprocess it, build the model, and start the training process.

4. **Model Evaluation**: After training, the model's performance is evaluated on the test set.

5. **Saving the Model**: The trained model is saved for future use.

## Data Format

The dataset must consist of `.npy` files containing the image data and labels:

- `data_train.npy`: Numpy array of image data.
- `labels_train.npy`: Numpy array of image labels.

## Hyperparameters and Model Configuration

This section provides detailed information about the hyperparameters used in the notebook and their role in the model's configuration.

### Data Preprocessing

1. **Data Splitting**
   - **Test Size (in `train_test_split`)**: Determines the proportion of the dataset to include in the test split. Default is set to `0.2` (20% of the data for testing).

2. **Image Resizing**
   - **Image Dimensions (in `image_batch_generator`)**: The target size to which all input images are resized. Set to `(300, 300)` by default, which is suitable for VGG16.

### Model Training

1. **Batch Size**
   - Used in `train_generator` and `validation_generator`. It determines the number of samples processed before the model is updated. Default value: `32`.

2. **Early Stopping (in `train_model` function)**
   - **Patience**: Number of epochs with no improvement after which training will be stopped. Set to `3` to prevent overfitting.

3. **Training Epochs**
   - The number of complete passes through the training dataset. The maximum is set to `100`, but training can stop early due to early stopping.

### VGG16 Model Configuration

1. **Pre-trained Weights**
   - The VGG16 model is initialized with weights pre-trained on ImageNet.

2. **Freezing Layers**
   - Layers in VGG16 are frozen up to the last 5 layers, meaning their weights will not be updated during training. This is a form of transfer learning.

3. **Custom Layers**
   - A `Flatten` layer to convert the 2D features to a 1D vector.
   - A `Dense` layer with `1024` units and `ReLU` activation.
   - `L2 Regularization` with a value of `0.01` in the Dense layer to reduce overfitting.
   - A `Dropout` layer with a rate of `0.5` to prevent overfitting.
   - The final `Dense` layer for classification with `9` units (number of classes) and `softmax` activation.

### Model Compilation

1. **Optimizer**: RMSprop with a learning rate of `0.0001`.
2. **Loss Function**: Categorical Crossentropy, suitable for multi-class classification tasks.
3. **Metrics**: Accuracy, to measure the performance of the model.

## Customization

Users can modify these hyperparameters according to their dataset and requirements. Adjustments can be made directly in the notebook cells where these parameters are defined.


You can modify parameters such as batch size, image dimensions, or model architecture by editing the corresponding cells in the notebook.

## Additional Notes

- Ensure that the dataset files (`data_train.npy` and `labels_train.npy`) are accessible from the notebook, preferably in the same directory for simplicity.
- The model architecture is based on VGG16, adapted for the ASL character classification task.
- The dataset is split into training, validation, and test sets.
- Early stopping is implemented during training to prevent overfitting.
- The notebook is structured to guide the user through the entire process of loading data, model building, training, evaluation,testing and saving the trained model for future use.

  Certainly! Here's an addition to the README including a section for the license. I'll use the MIT License as an example, as it's commonly used in open-source projects for its permissive nature. However, you should choose a license that best suits your project's needs.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-florida_men/blob/main/License.md) file for details.






