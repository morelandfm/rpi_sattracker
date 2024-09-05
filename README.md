Hi,<br/>
<br/>
My name is Jon Moreland, an applied physics undergrad in his final semester at the University of Arizona. This code is from a project that I was able to work on over the Summer in association with Marhold Space Systems. The goal for the project was to use a Raspberry Pi to identify and track a model satellite. Then meet a set of criteria for capture after identification (In this case proximity).<br/>

# Parts: <br/>

Raspberry Pi 4B 8GB<br/>
Pi Camera Module V2<br/>
Arducam IMX219 Wide Angle Camera Module<br/> 
Two 9G servos<br/>
Arduino Mega 2560 (You don't need this big of a board, just what I had on hand)<br/>
One Red LED<br/>
One Green LED<br/>
One small breadboard<br/>
TFMini Plus LiDAR sensor<br/>
Lots of wires<br/>
Raspberry Pi Case (Mine is 3D printed https://www.printables.com/search/models?q=raspberry%20pi%20case)<br/>
Pi Cam pan tilt stand (Mine is 3D printed https://www.thingiverse.com/thing:1401116)<br/>

# Initial setup<br/>

To start I needed to set up my Raspberry Pi and PiCam. There are plenty of guides for how to get started with a Raspberry Pi, and this project was more about me getting experience with different programs and operating systems so I just used Raspbian OS and Linux. If you are unfamiliar with getting started on the Raspberry Pi I recommend this video https://www.youtube.com/watch?v=rGygESilg8w. It's not required to go with the headless setup, but I recommend it just so you have that option in the future. I'm using tensorflow lite in this case due to the restricted capabilites of the raspberry pi.

From here you need to install all of the dependencies for the parts being used (i.e Pi Cam Module V2, tflite, opencv etc.) For all of the object detection portion of the project I followed this channels many tutorials https://www.youtube.com/@EdjeElectronics. Specifically I used https://www.youtube.com/watch?v=XZ7FYAMCc4M&t=1106s (Training a tensorflow model), https://www.youtube.com/watch?v=v0ssiOY6cfg (How to Capture and Label Training data), and https://www.youtube.com/watch?v=aimSGOAUI8Y&t=262s (How to Run TensorFlow Lite on Raspberry Pi for object detection). Some of the information is outdated since he was utilizing an older version of Raspbian OS, it's entirely possible that you are using a different Raspbian OS than I did while making this project. I recommend getting familiar with stackoverflow and googling solutions to your problems for fixes.

# Model training<br/>

Just a couple bits of advice if you are planning to train your own model and use the tutorials at the links provided. The quality of the model and it's ability to discern objects is related to the number of pictures used to train the model. In my case I used a little over 200 photos of the object that I wanted to detect, as you'll see it's a orange and yellow 3D printed model 6U cubesat that I had. 200 photos is very much on the low end of the amount that you would want to use for training a model. There was a point during testing where I had a blue shirt on a table that the program identified with 95% confidence was a model cubesat, so there is definitely some issues with the model in the github. However, it was enough for my purposes.<br/> 
### Quick tips:<br/>
-Take as many pictures as you can, the more the better within reason<br/>
-Use the same camera to take the pictures that you plan to use with the object detection model<br/>
-Use diverse environments for taking your pictures<br/>
-Don't put spaces in the names you use for the labels<br/>
-Definitely get the wide angle lens if you plan to use the Pi Cam Module V2<br/>
-I paid for the Google collab subscription, I felt like this helped since I couldn't watch the model for all of the time it was training and this helped avoid losing the data if the model finished training and I wasn't there<br/>
-Adjust the threshold as needed


After I had my model I used SatOD_Final.py to test the object detection program, you'll see in the notes that I modified the code used by Edje Electronics to implement the model. This is what I used to test confidence threshold changes, different styles of bounding boxes, and other changes to try and make the model implementation more efficient. All of this was done without a TPU and achieved between 4 and 5 fps. From what I've seen using a TPU you can get three to four times the amount of fps that I achieved here. If I get a chance to use one I'll edit this to detail my results.

# Communicating with arduino <br/>

Once the object detection model was working well I moved on to implementing the servos and the arduino with OD_Arduino_Final.py. There's a couple more dependencies that needed to be used for this which you'll see in the code, as well as some changes but similar implementation from the SatOD_Final.py. Here the idea is that we are taking the resolution value of the center of the bounding box generated around the detected object and sending that value to the arduino via serial communication. <br/>

### Initial Attempts and Issues <br/>

On my first attempt at getting this program to work I tried using the GPiO ports directly from the Raspberry Pi, without the additional use of the arduino. I'm not sure if it was incompetence on my part or maybe cheap servos, but I could not get the servos to work properly no matter how much tuning I did. From there I moved to using the arduino with serial communication. When setting up serial communication there's a few steps you have to go through to make sure that the serial data is being sent to the correct port. That's going to differ depending on what machine you're using but is a pretty straightforward process (https://www.youtube.com/watch?v=xc9rUI0F6Iw). What's important here is that the baud rate on the board and the program are the same. If you use the code in this repository you will see they are set to 115200.

Another issue was the constant changing in bounding box size. When the object detection is running it's creating a bounding box around the object that it is detecting, and as that object gets closer to or further away from the camera that box changes size. The information being sent to the arduino is with respect to the center of the bounding box, and if that center value is constantly changing due to the programs interpretation of the size of the object it's detecting then the servos will continue to adjust a lot. This causes a lot of jittter in the movement, and since I was trying for a much smoother tracking, in order to keep this from happening I made the bounding box size constant, this way the value for the center of the box was constant with respect to the box itself. Now the program just needs to determing where the center of the box is with respect to the resolution of the camera.

Here I'll add that it's important to make adjustments to the code if you plan on using a different resolution than I used. It's not too hard to determine or set the resolution that the camera will be using, but it's important that the resolution for the video is the same as what's input in the arduino code with respect to the servo mapping. 

Another issue is that the only way to correctly track an object detected by the program is for there to only be ONE object for the program to correctly detect. Since the program can only center on one object, if the program detects a second object it will throw the program off when it detects the second object. I didn't spend a ton of time trying to solve this problem other than raising the threshold and using only one test object for the program. But the little research I did didn't look like it was possible to set the program to only track one object. My understanding is that the object detection model is running on every frame individually, and so there is no way for the program to stop looking for new objects after spotting the first because there is no way for the model to know which is first in between frames and then stop looking.

If you put everything together, run the program, and the servos are turning away from the object the camera detected reverse the angle mapping on the servos. You'll see in the arduino code that I have one servo mapped from 0, 180 and one from 180, 0 and that's due to the fact that how my pan tilt is setup I had to put one of the servos in reverse so I had to switch the mapping on that specific servo.
