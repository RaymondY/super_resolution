# super_resolution
A PyTorch implementation of final project for EE 541

# Environment
- Python 3.8
- PyTorch Preview (Nightly) for MPS acceleration is available on MacOS 12.3+
- PyTorch 1.13.0 (CUDA 10.2) 
- torchvision 0.14.0
- tqdm 4.64.1
- kornia 0.6.8

# Files
- `main.py`: main file to run the program
- `train_test.py`: train and test functions
- `model.py`: model definition
- `utils.py`: utility functions
- `config.py`: configuration file
- `sup.py`: supplementary functions for computing mean and standard deviation of the training set and validation set
- `/data`: data folder; Put the data according to paths in the configuration file.
- `/model`: model folder; Trained models will be saved here, named by the current time.

# How to run
- `python main.py`: train and test the model. You can adjust the number of residual blocks in main.py. Run `read_model_and_test()` to test the saved model. Adjust index in `read_model_and_test()` to visualize the results of different images.