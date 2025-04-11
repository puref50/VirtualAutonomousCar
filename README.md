# VirtualAutonomousCar
## Project Structure
- train_environment.py
- train.py
- test_environment.py
- test.py
- train_CNN.ipynb
- generate_image.ipynb
- CNN_image_model.h5
## Setting Up
Install:
-python==3.9
-carla==0.9.15
-gymnasium==0.29.1
-keras==2.10.0
-matplotlib==3.9.4
-numpy==1.26.4
-opencv-python==4.10.0.84
-pygame==2.6.1
-stable_baselines3==2.4.0
-tensorflow==2.10.1
-torch==2.5.1
## Execution
-Run generate_image.ipynb to collect training data first.
-Run train_CNN.ipynb to get the trained CNN model
-Run train.py to do reinforcement learning
-Run test.py to see how the car drives automatically
