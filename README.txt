README
I trained an object detection model using the TensorFlow object detection API, then I extracted the frozen graph training data for use in an object detection program.
Dependencies:
•	TensorFlow 1.15
•	Python 3.5 or higher
•	Pillow 
•	Matplotlib
•	OpenCV
•	NumPy
•	TensorFlow Object detection API
•	Tkinter
Installation:
1.	Go to  https://www.python.org/downloads/release/python-377/, scroll down to Files, and download python 3.7.
(if you have a 32 bit version of windows install the windows x86 executable installer and if you have a 64 bit version of windows install the windows x86-64 executable installer)
2.	Go to your Downloads Folder and click on the downloaded file and follow the instructions (be sure to click the add python to PATH box when installing)
3.	From the command line (press the windows key, then type cmd and press enter)
4.	Type: python -m pip install pillow and press enter
5.	Type: python -m pip install numpy and press enter
6.	Type: python -m pip install opencv-python and press enter
7.	Type: python -m pip install tensorflow==1.15 and press enter
8.	Type: python -m pip install matplotlib and press enter
9.	Unzip the zip folder titled condors and move the folder into your documents folder. 
Utilization:
•	Go into the condor detection folder that should now be located in your documents folder
•	Find the file titled condordetection.py
•	Double click that file, be patient it may take a second or two. You should now see a live feed of your condors, with boxes around them identifying who is who!
•	Now there should be a file called condorlist.csv, that is a record of the condors that the camera saw that day. You should be able to open this file in excel!

I used tkinter to make a GUI that allows the user to start and stop the program. The program intakes live video feed from a camera using open cv and then uses the data from the frozen graph to detect and count the number of distinct objects it sees.
