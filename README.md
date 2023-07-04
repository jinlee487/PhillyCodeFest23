# ar_troubleshooter
Use AR to help customers perform simple troubleshooting steps to improve their internet connectivity. Judging criteria includes: Creativity of solution
Thoroughness of implementation details
Simplicity of solution

<img width="310" alt="image" src="https://github.com/jinlee487/PhillyCodeFest23/assets/46912607/2350d67a-4244-4cd2-90ef-7adb585be3d3">

## Inspiration
Troubleshooting a broken internet connection can be stressful. Following written instructions can be especially challenging for those who don't speak/read the language and/or those who aren't familiar with technical terms (e.g., what is the coax cable? what does the modem look like?). **AI technology** can enable customers to simply interact with **Augmented Reality (AR)** and make the entire troubleshooting process easier to follow and more pleasant. 

## What it does
We built an AR application in which customers can troubleshoot Internet problems by simply clicking what they see on their cameras. Among various troubleshooting scenarios, for Philly CodeFest 2023, we decided to focus on one general problem that many Internet users may have faced: self-install the modem. 

Meet Jay, who just got his Getting Started Kit from Comcast delivered, but is very much frustrated because he doesn't understand any of the technical terms in the installation guide. Here is what will happen to help reduce cognitive loads for Jay and make his experience more fun and pleasant.

**1. Scan the kit.**

Jay will simply turn on the camera on his smartphone and scan what is in the Getting Started Kit. 

**2.  Recognize each item.**

Our machine learning (ML) model will be able to recognize the modem, Coax cable, ethernet cable, and power cord. 

**3. Pick up the correct item.**

Our AR application will guide Jay to pick up the correct gadget without looking into the instruction book.

**4. Plug into the correct input source.**

Our model will recognize input sources on the modem and guide Jay to plug the right cable into each input. 

**5. The internet is set up!**

## How we built it
### 1) Model
**Step 1. Data Collection**

**Step 2. Data Processing**

![Image 3-12-23 at 11 38 AM](https://user-images.githubusercontent.com/46912607/224555448-beb3df23-bb5a-40f1-a715-6ff07832c6a2.jpeg)

**Step 3. Data Training/Testing** 

<img src="https://user-images.githubusercontent.com/46912607/224554511-dfbf2a84-8d5d-4f74-808d-7684e12e775a.jpeg" width="50" height="50"/>


### 2) AR application
We built **a Flask web application** that streams processed video to the **frontend**. 

<img width="1470" alt="image" src="https://user-images.githubusercontent.com/46912607/224555864-7081a24d-3b9e-4988-a32f-012bf222b8d1.png">

## Challenges we ran into
* Data processing was long hours of work. 
* We wanted to create a socket for video streaming but we only had limited time to make it work.

## Accomplishments that we're proud of
* Our AR-Troubleshooter can recognize the trained items. 
* We made a web application that is able to stream the ML model-processed video to the frontend.

## What we learned
We have to optimize our time depending on the timeline of the project and have a better estimation of how long it will take to finish each task. Preparing data for object recognition modeling and training the model itself can take hours of work. These processes may have been facilitated if we had more personnel to divide the work. 

## What's next?
* We want to collect more data to improve the model performance. 
* We would like to apply our improved model to more extensive troubleshooting scenarios and examine how applicable our AR technology is to providing troubleshooting solutions for various real-world problems. 
* Plus, we would like to add various sensory feedback when the customer follows each troubleshooting step to make the technology more inclusive (e.g., phone vibrates when a customer with hearing problems picks up the correct item). 

## Procedure to run the app:

```
python -m venv venv
```

reload your terminal so that it runs the virtualenv

```
pip install -r "requirements.txt"
```

you can run either the infer-simple.py or the ar_troubleshooter.py


## Reference:
--[Roboflow tutorial](https://blog.roboflow.com/python-webcam/)

--[Camera App with Flask and OpenCV Tutorial](https://naghemanth.medium.com/camera-app-with-flask-and-opencv-bd147f6c0eec?source=friends_link&sk=705255bd58cf139ad95ab2149806d8c6)

--[cvzone](https://github.com/cvzone/cvzone)

## Demo:
--[Philly Codefest 23](https://devpost.com/software/video-stream-app)
