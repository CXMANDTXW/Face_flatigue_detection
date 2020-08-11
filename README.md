# Face_flatigue_detection
It's a simple script for face flatigue detection for Windows.
Including blink,Yawn,Nod detection.

# Intallation process

## step 1:
 Install all libraries 
- scipy  (pip install scipy)
 
- OpenCv

- imutils (pip install imutils)

- dlib

- python


# installation of Dlib libary 
These instructions assume you are on Windows.

Pre-reqs:
Have Python 3 installed. We need to keep the python version consistent with the Dlib version

- On Linux:
  On Linux, the installation of Dlib is much easier than that of windows.You can search Google for installation methods or refer to the official installation manual.
  http://dlib.net/compile.html
  
- On Windows:
  For py3.7, you can download the dlib file in https://pan.baidu.com/s/1MRCDNF4ha1-cLO_oGYqXDQ  提取码：afq1
  For other version,you can find the suitable file in https://pypi.org/simple/dlib/  
  Then type in command line:
  ```
  pip install XXXXX.whl
  ```

# step 1
- We need download the pretrained file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  61M
- Make sure the pre training files and face_detection.py are in the same directory

# step 2
For real-time detection(USB Camera).Do not use -v to specify a video file.
```
python face_detection.py -p shape_predictor_68_face_landmarks.dat

```
For video detection
```
python face_detection.py -p shape_predictor_68_face_landmarks.dat -v your_video.mp4
```


