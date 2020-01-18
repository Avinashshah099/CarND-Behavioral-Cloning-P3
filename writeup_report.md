# **Behavioral Cloning**

## Writeup

### Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/left_2019_05_12_22_34_15_157.jpg "Track1 Left Camera"
[image3]: ./images/center_2019_05_12_22_34_15_157.jpg "Track1 Center Camera"
[image4]: ./images/right_2019_05_12_22_34_15_157.jpg "Track1 Right Camera"
[image5]: ./images/left_2019_05_19_18_08_24_634.jpg "Track2 Left Camera"
[image6]: ./images/center_2019_05_19_18_08_24_634.jpg "Track2 Center Camera"
[image7]: ./images/right_2019_05_19_18_08_24_634.jpg "Track2 Right Camera"
[image8]: ./images/center_2019_05_12_22_34_15_157_FlipH.jpg "Track1 Horizontal Flip"
[image9]: ./images/center_2019_05_19_18_08_24_634_FlipH.jpg "Track2 Horizontal Flip"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [normalization.py](./normalization.py) for defining normalization cutom layer
* [model.h5](./mode.h5) containing a trained convolution neural network
* [writeup_report.md](./writeup_report.md) summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:

```bash
python3 drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 ([model.py](./model.py) lines 124-128)

The model includes RELU layers to introduce nonlinearity (code line 124-128), and the data is normalized in the model using custom Keras layer (code line 121).

Custom Normalization layer was introduced as a replacement for Keras Lambda layer in order to enable saving and reading model.h5 on different platforms.

#### 2. Attempts to reduce overfitting in the model

Large data sets enabled to use model without dropout layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 138-140). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py](./model.py) line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. It was sufficient to use center lane driving data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolution neural network.

My first step was to use a convolution neural network model similar to the LeNet-5. I thought this model might be appropriate because it is relatively simple and yet powerful.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I replaced the model with more powerful architecture from [nVidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

Then I tunned the model architecture to fit project requirements.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track1. Surprisingly it did not fell off the track2 which I considered to be much more difficult to learn. To improve the driving behavior on the track1 I have doubled the track1 training data set to meet the size of the track2 training data set (track2 is twice as long as track1). Unfortunately it did not help. At the end I realized that although track2 is more difficult learn than track1, track2 has simpler color scheme (it consists mostly from green and gray colors). I figured out that I forgot to convert images from BGR to RGB before training.

After implementing simple fix, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture ([model.py](./model.py) lines 124-133) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 12 laps on track1 (6 laps in each direction) and 6 laps on track2 (3 laps in each direction) using center lane driving. Here is an example images of center lane driving on both tracks:

| Left Camera         | Center Camera       | Right Camera        |
|:-------------------:|:-------------------:|:-------------------:|
| ![alt text][image2] | ![alt text][image3] | ![alt text][image4] |
| ![alt text][image5] | ![alt text][image6] | ![alt text][image7] |

To augment the data set, I also flipped images and angles thinking that this would enforce model to better generalize features. For example, here are images that has then been flipped:

| Normal              | Horizontally Flipped |
|:-------------------:|:--------------------:|
| ![alt text][image3] | ![alt text][image8]  |
| ![alt text][image6] | ![alt text][image9]  |

After the collection process, I had 170304 number of data points. I then preprocessed this data by cropping the images by (55,25) pixes from (top,bottom) and normalizing with formula `x/127.5 - 1.0`.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-5 as evidenced in [model.log](./model.log) file. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Support of Intel® OpenVINO™ Inference Engine

#### 1. Converting Keras model to TensorFlow frozen protobuf

In order to be able use pre-trained model in OpenVINO it is required to convert it to one of the supported model formats.
Having pre-trained Keras (`.h5`) model in place I decided to convert it to TensorFlow frozen protobuf (`.pb`)

```bash
> ipython3
```

```python
import drive
drive.export_keras_to_tf('model.h5', 'model.pb')
```

#### 2. Converting TensorFlow frozen protobuf to Intermediate Representation using Model Optimizer

My model has Cropping2D and Normalization preprocessing layers that are not supported by Model Optimizer.
One of the methods to solve the problem is to cutoff preprocessing layers from the model while converting it to Intermediate Representation.
In order to be able to use IR with VPU (and CPU) it is required to use FP16 data type.

```bash
> mo_tf.py -m model.pb -b 1 --input=conv2d_1/convolution --data_type=FP16
```

#### 3. Using Intel® OpenVINO™ Inference Engine

In order to be able to use IR of the model with Inference Engine is necessary to preprocess the images before passing it to the `infer` function.

#### 4. Making Predictions in Real-Time for Images Generated by Simulator

Run Udacity Simulator and execute:

```python
> python3 drive.py model.xml
```

`drive.py` script first try to use NCS2 if available. It uses CPU inference as a fallback.

#### 5. Results

The network architecture is based on LeNet-5 network which is fairly simple thus I do not expect any significant boost (if any) when processing it on VPU over CPU.
In order to make fair comparison of model inference on both CPU and NCS2 I used:

```bash
benchmark_app -m model.xml -d [CPU|MYRIAD]
```

The hypothesis was confirmed by following result:

```text
i5-3360m CPU:
Count:      29404 iterations
Duration:   60011.86 ms
Latency:    8.08 ms
Throughput: 489.97 FPS
```

```text
NCS2 MYRIAD VPU:
Count:      27512 iterations
Duration:   60009.71 ms
Latency:    8.63 ms
Throughput: 458.46 FPS
```

Most likely the results would look significantly different if I use network with dozens or hundreds of layers.
