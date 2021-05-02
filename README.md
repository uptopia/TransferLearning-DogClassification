```
<Terminal 1>
    $roscore

<Terminal 2> [Mode 1] input image from file
    NONE
OR
<Terminal 2> [Mode 2] usb_cam
    $cd <usb_cam_ws>
    $roslaunch usb_cam usb_cam-test.launch video_device:=/dev/video0
OR
<Terminal 2> [Mode 3] Realsense D435i
    $cd <realsense-ros_ws>
    $roslaunch realsense2_camera rs_camera.launch
 
<Terminal 3>
    $cd <TransferLearning-DogClassification_ws>
    $. devel/setup.bash
    $rosrun publish_img publish_img.py

<Terminal 4>
    $cd <TransferLearning-DogClassification_ws>
    $. devel/setup.bash
    $rosrun classify_dogs dog_predict.py
```

![Python 3.6.9](https://img.shields.io/badge/python-3.6.9-green.svg)
# Transfer Learning for Dog Classification

<!-- [TODO] This is the official DOPE ROS package for detection and 6-DoF pose estimation of **known objects** from an RGB camera.  The network has been trained on the following YCB objects:  cracker box, sugar box, tomato soup can, mustard bottle, potted meat can, and gelatin box.  For more details, see our [CoRL 2018 paper](https://arxiv.org/abs/1809.10790) and [video](https://youtu.be/yVGViBqWtBI). -->

[ppt](https://arxiv.org/abs/1809.10790)
[video](https://youtu.be/yVGViBqWtBI)

*Note:*  The instructions below refer to inference only.  Training code is also provided but not supported.

![Classification Model](dog_classification_model.png) #4 classes of dogs
![Dog Inference Results](dog_inference_result.png)


## Installing
##Dependencies
software environment
ubuntu 18.04
python 3.6.9
pytorch 1.8.1+cpu
opencv 4.5.1 cv2
ROS melodic
cv_bridge (自己編譯)
realsense sdk
realsense_ros

GPU

<!-- [TODO]We have tested on Ubuntu 16.04 and 18.04 with ROS Kinetic and Lunar with an NVIDIA Titan X and RTX 2080ti with python 2.7.  The code may work on other systems. -->

<!-- [TODO]The following steps describe the native installation. Alternatively, use the provided [Docker image](docker/readme.md) and skip to Step #7. -->
[Docker image]

setup
0.download traning data
1.download weights
2.dependencies

1. **Install ROS**

    <!-- Follow these [instructions](http://wiki.ros.org/kinetic/Installation/Ubuntu).
    You can select any of the default configurations in step 1.4; even the
    ROS-Base (Bare Bones) package (`ros-kinetic-ros-base`) is enough. -->

2. **Create a catkin workspace** (if you do not already have one). To create a catkin workspace, follow these [instructions](http://wiki.ros.org/catkin/Tutorials/create_a_workspace):
    ```
    $ mkdir -p ~/catkin_ws/src   # Replace `catkin_ws` with the name of your workspace
    $ cd ~/catkin_ws/
    $ catkin_make
    ```

3. **Clone this repository**
    ```
    $ cd ~/catkin_ws/src
    $ git clone https://github.com/uptopia/TransferLearning-DogClassification.git
    ```
    structure of package
    ![TransferLearning-DogClassification_tree](dogclassification_tree.png)

4. **Install python dependencies**
    Python packages:

    ```
    $ cd ~/catkin_ws/src/TransferLearning-DogClassification
    $ pip3 install -r requirements.txt
    ```
    <!-- pyrr==0.9.2
    torch==0.4.0
    numpy==1.14.2
    scipy==1.1.0
    opencv_python==3.4.1.15
    Pillow==5.3.0
    torchvision==0.2.1 -->

5. **Install ROS dependencies**
    <!-- ```
    $ cd ~/catkin_ws
    $ rosdep install --from-paths src -i --rosdistro kinetic
    $ sudo apt-get install ros-kinetic-rosbash ros-kinetic-ros-comm
    ``` -->
    ROS Packages:
    <!-- $ sudo apt-get install ros-<distro>-pcl-ros
    $ sudo apt-get install ros-<distro>-camera-info-manager
    $ sudo apt-get install ros-<distro>-position-controllers
    $ sudo apt-get install ros-<distro>-velocity-controllers
    $ sudo apt-get install ros-<distro>-effort-controllers
    $ sudo apt-get install ros-<distro>-joint-state-controller
    $ sudo apt-get install ros-<distro>-joint-state-publisher
    $ sudo apt-get install ros-<distro>-aruco-detect
    $ sudo apt-get install ros-<distro>-visp-hand2eye-calibration -->

6. **Build**
    ```
    $ cd ~/catkin_ws
    $ catkin_make
    ```

7. **Download [the weights](https://drive.google.com/drive/folders/19FlH4pgi4C8GCmsGQPueILWLrZO25fvJ?usp=sharing)** and save them to the `weights` folder, *i.e.*, `~/catkin_ws/src/TransferLearning-DogClassification/weights/`.

## Training the network
* There are 2 methods to train the transfer learning network.
   * **Method 1** train using ipynb [train.ipynb](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg)

   * **Method 2** train using local computer
       ```    
       $ cd ~/catkin_ws/src/TransferLearning-DogClassification/src
       $ python3 train.py
       ```
       save weights in `weights` folder

## Using the trained network

1. **Start ROS master** @Terminal 1
    ```
    $ cd ~/catkin_ws
    $ source devel/setup.bash
    $ roscore
    ```

2. **Start camera node** @Terminal 2 (or start your own camera node)
   * **Mode 1**: input image from path
        * Camera is not needed. Skip this step!
        <br>

   * **Mode 2**: usb_cam 
        * [ROS usb_cam Installation](https://blog.csdn.net/dengheCSDN/article/details/78983993)
        * Modify `usb_cam-test.launch` ($cd <usb_cam_ws>/src/usb_cam/launch)
            i.e. [USB2.0 VGA UVC WebCam](https://webcamtests.com/reviews/2461)
            &emsp;&ensp;Edit config info `framerate` from 30 to 8: `<param name="framerate" value="8" />`     
        * Publishes RGB images to `/usb_cam/image_raw`
            ```
            $ cd <usb_cam_ws>
            $. devel/setup.bash
            $roslaunch usb_cam usb_cam-test.launch video_device:=/dev/video0
            ```
     
   * **Mode 3**: Realsense-D435i       
        * [realsense-ros Installation](https://github.com/IntelRealSense/realsense-ros)
        * Publishes RGB images to `/camera/color/image_raw`
            ```
            $ cd <realsense_ros_ws>
            $. devel/setup.bash
            $roslaunch realsense2_camera rs_camera.launch
            ```

3. **Start publish_img node** @Terminal 3
    ```
    $cd <TransferLearning-DogClassification>
    $. devel/setup.bash
    $rosrun publish_img publish_img.py
    ```
   * **Mode 1**: input image from path
   * **Mode 2**: usb_cam
   * **Mode 3**: Realsense-D435i 
    <br>
    
4. **Start dog_predict node** @Terminal 4
    ```
    $cd <TransferLearning-DogClassification>
    $. devel/setup.bash
    $rosrun classify_dogs dog_predict.py
    ```

## ROS Topics
* The following ROS topics are published:
    ```
    /classify_dogs/publish_img  # RGB images from camera
    /classify_dogs/dog_class    # dimensions of object
    ```
