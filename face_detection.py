# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import winsound
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
def mouth_aspect_ratio(mouth):
	A=dist.euclidean(mouth[3],mouth[9])
	B=dist.euclidean(mouth[0],mouth[6])
	return A/B
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
BINK_WARN_TIME = 1000
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
EYE_ALTER_FRAMES = 100
MOUTH_THRESH = 1.0
MOUTH_CONSEC_FRAMES = 15
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
M_COUNTER = 0
M_TOTAL = 0
N_TOTAL=0
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
time.sleep(1.0)
# loop over frames from the video stream
k=0
status='Good'
duration = 1000
freq = 1000
n_pre=1000
n_number=0
while True:

	frame = vs.read()
	n_number=(n_number+1)%31
	k = (k+1)%(BINK_WARN_TIME+1)
	if k==BINK_WARN_TIME:
		status = 'Good'
		TOTAL,M_TOTAL,N_TOTAL=0,0,0
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		nose = shape[nStart:nEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mar = mouth_aspect_ratio(mouth)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		noseHull = cv2.convexHull(nose)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

		if mar > MOUTH_THRESH:
			M_COUNTER+=1
		else:
			if M_COUNTER >= MOUTH_CONSEC_FRAMES:
				M_TOTAL+=1
			M_COUNTER = 0
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_ALTER_FRAMES:
				TOTAL,k=0,0
				status = 'Fatigue'
				winsound.Beep(freq, duration)  # 调用喇叭，设置声音大小，与时间长短
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			COUNTER = 0
		if TOTAL>=15 or M_TOTAL>=4 or N_TOTAL>=5:
			k, TOTAL, M_TOTAL, N_TOTAL = 0, 0, 0, 0
			status = 'Fatigue'
			winsound.Beep(freq, duration)  # 调用喇叭，设置声音大小，与时间长短

		local_nose=0
		nose=nose[4:]
		for i in range(len(nose)):
			local_nose+=nose[i][1]
		local_nose/=len(nose)
		if n_number==30:
			N_Dis = local_nose - n_pre
			if N_Dis>20:
				N_TOTAL+=1
			n_pre=local_nose
		# if N_Dis>=2:
		# 	n_number+=1
		# else:n_number=0
		# if n_number==2:
		# 	N_TOTAL+=1
		# 	n_number=0
		# n_pre=local_nose
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		cv2.putText(frame, "Yawn: {}".format(M_TOTAL), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		cv2.putText(frame, "Nod: {}".format(N_TOTAL), (10, 100),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		# cv2.putText(frame, "local: {:.2f}".format(N_Dis), (300, 100),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		if status=='Good':

			cv2.putText(frame, "{}".format(status), (10, 150),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		else:
			cv2.putText(frame, "{}".format(status), (10, 150),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()