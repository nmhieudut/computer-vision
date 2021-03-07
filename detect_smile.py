# import packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

# construct agrument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required = True,
    help = "path to where the face cascade resides")
ap.add_argument("-m", "--model", required = True,
    help = "path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])
smile_count = 0
not_smile_count = 0
total_frame=0
count = 0

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('./input/huy.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    total_frame+=1
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # reszie the frame, convert it to grayscale, then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width = 600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so
    # that we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
        minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via CNN
        roi = gray[fY: fY + fH, fX: fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        # determine the probabilities of both "smiling" and "not similing"
        # then set the label accordingly
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        if label == "Smiling":
            smile_count += 1
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)
            cv2.putText(frame, 'smile:  '+str(smile_count), (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 21), 2)
            cv2.imwrite("./output/positive/frame%d.jpg" % count, frame)
            ret, frame = cap.read()
            print('Read a new frame: ', ret)
            count += 1 # save frame as JPEG file 
        else:
            not_smile_count += 1
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)
            cv2.putText(frame, 'not smile:  '+str(not_smile_count), (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imwrite("./output/negative/frame%d.jpg" % count, frame)     # save frame as JPEG file      
            ret, frame = cap.read()
            count += 1
        # display the label and bounding box rectangle on the output frame
        # cv2.putText(frameClone, label, (fX, fY - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
        #     (0, 0, 255), 2)
        # cv2.putText(frameClone, 'smile:  '+str(smile_count), (0, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 21), 2)
        # cv2.putText(frameClone, 'not smile:  '+str(not_smile_count), (0, 60),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # cv2.putText(frameClone, 'total frame:  '+str(total_frame), (0, 90),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 234), 2)
        # cv2.putText(frameClone, 'detected frame:  '+str(smile_count+not_smile_count), (0, 120),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 98, 0), 2)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)
  
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
print('smile count: ',smile_count)
print('not smile count: ',not_smile_count)
