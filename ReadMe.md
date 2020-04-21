# Social Distancing Tracker using OpenCV

## Introduction:
    A simple Social Distancing Tracker using OpenCV
    - People detection is done using HOG descriptor. Because it don't need heavy processing power 
    and it is better than Haar Cascade Classifier.
    - You can use faster RCNN deep learning model for more accurate people detection (Computationally heavy).
    - Determined the centroid of people.
    - Determined the distance of the indiviuals with each other.
    - if they are too close then a red line is drawn else if they are a little bit close then yellow
    line is drawn.
## Requirements:
    - opencv-python
    - numpy
    - imutils
  
## To Run:
    - python main.py
    
## Challenges:
    - Perspective vision is a challenge for determining the threshold distance for closeness.
    - Can be improved with more processing power and high end deep learning models.
