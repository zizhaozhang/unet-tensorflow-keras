# Unet-tensorflow-keras
A concise code for training and evaluating Unet using tensorflow+keras 

A simple practice of the mixture usage of tensorflow and keras for the segmentation task. 
Sometime using Keras to manage the training is not flexiable. But we still want to utilize the convenience of Keras to build the model.
Using Keras to build the model is super easy and fully compatible with Tensorflow. See https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html for an introduction. 

I use the Unet architecture and modify its unsampling part to automatically adjust the feature map width and height when merge (concat) with previous layers. In this way, we do not need to compute the specific input size to fit the model but take an arbitrary size. 

## Warnings
If you use a recent version of Keras, there is a bug of synchronizing images and labels in dataLoader, which makes the loaded image and label are not paired even you set the same seed. You need to slightly modify the keras/preprocessing/image.py follwing [this](https://github.com/fchollet/keras/commit/419105bd371460720332dcacee0b681322375e9e#diff-93850fa46a789f2e5905894ad0e7bee4).

**UPDATE, July 2017**: 
  - Change the code to fit Tensorflow > 1.0 
  - Adding a loader file to use Keras generator to load image and mask with automatic augmentation
  - Adding a VIS module to manage the evaluation metric.
  - Adding opt.py to support easier use

### Usage
- See loader.py to organize your train/test data hierarchy 
- Set necessary hpyerparameters and run train.py 

  ```bash
  python train.py --data_path ./datasets/your_dataset_folder/ --checkpoint_path ./checkpoints/unet_example/
  ``` 
- Visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard

  ```
  tensorboard --logdir=train_log/
  ``` 
- When checkpoints are saved, you can use eval.py to test an input image with an arbitrary size.

- Evaluate your model
  ```bash
  python eval.py --data_path ./datasets/your_dataset_folder/ --load_from_checkpoint ./checkpoints/unet_example/model-0 --batch_size 1
  ```
