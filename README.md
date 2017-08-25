# NoScope

This is a fork from the offical Noscope. We added a few notes about the issues we had during the install and solutions. Please refer to the official Noscope [github](https://github.com/stanford-futuredata/noscope), [blog post](http://dawn.cs.stanford.edu/2017/06/22/noscope/) and [paper](https://arxiv.org/abs/1703.02529)  for mroe details 

Please read Update section for the latest update

## Requirements

This repository contains the code for the optimization step in the paper. The inference code is
[here](https://github.com/stanford-futuredata/tensorflow-noscope/tree/speedhax).

You will need the following installed:
- python-setuptools python-tk
- CUDA, CUDNN
- tensorflow-gpu, OpenCV 3.2 with FFmpeg bindings
- g++ 5.4 or later

Your machine will need at least:
- AVX2 capabilities 
- 300+GB of memory 
- 500+GB of space
- A GPU (this has only been tested with NVIDIA K80 and P100)


## Setting up the inference engine

1. To set up the inference engine, do the following:
```
git clone https://github.com/stanford-futuredata/tensorflow-noscope.git
cd tensorflow-noscope
git checkout speedhax
git submodule init
git submodule update
```
The build will fail. To fix this, update the BUILD file to point towards your OpenCV install and add
this directory to your PATH environmental variable. Please encourage the Tensorflow developers to
support non-bazel building and linking. Due to a quirk in bazel, it may occasionally "forget" that
tensorflow-noscope was built. If this happens, rebuild.


2. Configure the interface

Install a C++ compliler and make sure it is the version is >5.4 
```
pip install g++
```

Start configuration
```
./configure
```

After you start configuration, you are required to setup the interface. This is the setup I am using,
```
Please specify the location of python. [Default is /usr/bin/python]: // Press Enter
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: // Press Enter
Do you wish to use jemalloc as the malloc implementation? [Y/n] // Press Enter
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] // Press Enter
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] // Press Enter
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] // Press Enter
No XLA support will be enabled for TensorFlow
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
  /home/ubuntu/src/cntk/bindings/python
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages] // Press Enter

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with OpenCL support? [y/N] // Press Enter
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] // use CUDA, Y and Press Enter
CUDA support will be enabled for TensorFlow
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: // Press Enter
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: // Press Enter
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify the Cudnn version you want to use. [Leave empty to use system default]: // Press Enter
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: // Press Enter
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: "3.5,5.2"]: // Press Enter
```


3. Then continue to build
```
cd tensorflow
bazel build -c opt --copt=-mavx2 --config=cuda noscope
```


## Running the example

Once you have inference engine set up, the `example/` subfolder contains the script to reproduce
Figure 5d in the paper.

1. Download the coral-reef video and labels:
```
wget https://storage.googleapis.com/noscope-data/csvs-yolo/coral-reef-long.csv
wget https://storage.googleapis.com/noscope-data/videos/coral-reef-long.mp4
```
or 
```
wget https://storage.googleapis.com/noscope-data/csvs-yolo/jackson-town-squarecsv
wget https://storage.googleapis.com/noscope-data/videos/jackson-town-square.mp4
```


2. If the example video is too large, I have written a mini video-cutter under the main folder. The program will look for the video unders the data folder and save the new video in the same folder. My file structure as below
```
noscope
    |--data
    |   |-cnn-avg
    |   |-cnn-models
    |   |-csv
    |   |-experiments
    |   |-videocut.py
    |   |-videos
    |--main
    |--tensorflow-noscope
```


Usage:
```
python videocut.py --num_of_frame 3000 --vname jackson-town-square --out_vname sm-jackson-town-square 
```


3. Before running you will need to update run.sh and noscope_motherdog.py under the example folder. 
Update run.sh
```
CODE_DIR="/home/ubuntu/noscope/main"    #Update the path
DATA_DIR="/home/ubuntu/noscope/data"    #Update the path

VIDEO_NAME="jackson-town-square"
OBJECT="car"
NUM_FRAMES="918000"                     #Update the number of frames if your video is shortened
START_FRAME="0"
GPU_NUM="0"
```

Update the noscope_motherdog.py
```
time {tf_prefix}/bazel-bin/tensorflow/noscope/noscope \\            #Update the path. {tf_prefix} is your CODE_DIR

    --yolo_cfg=%s/home/ubuntu/darknet/cfg/yolo.cfg \\               #Update the path to your YOLO config file
    --yolo_weights=%s/home/ubuntu/darknet/weights/yolo.weights \\   #Update the path to your YOLO weights file
```

Please note that the original code use CV2 to extract frame however this video capture function is not working in my machine. I modify the VideoUtils.py under the exp/noscope to use skvideo.io 
```
pip install sk-video 
```

4. Run 
```
./run.sh
```


## Update
25/08/2017 - We updated the GPU which it support AVX2. However we do not have enough memory to get it working. From my observation it stopped when it tries to build a YOLO weight file. The requirement of this program is inreasonable (it requires 300GB RAM). We decide not to upgrade the memory.




