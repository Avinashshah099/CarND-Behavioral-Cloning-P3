# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains Behavioral Cloning Project.

This project enables to clone driving behavior using Keras. The model will output a steering angle to an autonomous vehicle.

Data about driving behavior can be collected using [simulator](https://github.com/udacity/self-driving-car-sim#term-1) where car can be steered around two tracks. Collected image data and steering angles can be used to train a neural network and then the model can be used to drive the car autonomously around the tracks.

The project consists of following files: 
* [model.py](model.py) (script used to create and train the model)
* [normalization.py](normalization.py) (supplementary script that defines custom Keras normalization layer)
* [drive.py](drive.py) (script to drive the car - feel free to modify this file)
* [model.h5](model.h5) (a trained Keras model)

## Details About Files In This Directory

### `model.py`

Once driving data are collected and stored in `data` directory this script can be used to train the model.
```bash
> python3 model.py
``` 
As a result this script creates `model.h5` file with the model.

### `drive.py`

Saved `model.h5` model can be used with drive.py to make predictions on individual images in real-time and send the predicted angle back to the simulator server via a websocket connection:

```bash
> python3 drive.py model.h5
```

**Note**: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```bash
> python3 drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```bash
> python3 video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Model Architecture and Training Documentation

Please refer to [writeup](writeup_report.md) for implementation details.

## License

MIT License

Copyright (c) 2019 Marcin Sielski