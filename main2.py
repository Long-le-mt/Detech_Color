# Python code for Multiple Color Detection


import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):
	
	# Reading the video from the
	# webcam in image frames
	_, imageFrame = webcam.read()

	# Convert the imageFrame in
	# BGR(RGB color space) to
	# HSV(hue-saturation-value)
	# color space
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)


# CREATE MASK OF COLOR

	# Set range for red color and
	# define mask
	white_lower = np.array([70, 10, 130], np.uint8)
	white_upper = np.array([180, 110, 255], np.uint8)
	white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

	# Set range for red color and
	# define mask
	red_lower = np.array([160, 90, 100], np.uint8)
	red_upper = np.array([180, 175, 165], np.uint8)
	red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for orange color and
	# define mask
	orange_lower = np.array([0,  140,  196], np.uint8)
	orange_upper = np.array([26,  255,  255], np.uint8)
	orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	# Set range for green color and
	# define mask
	green_lower = np.array([42, 100, 100], np.uint8)
	green_upper = np.array([75, 255, 255], np.uint8)
	green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

	# Set range for blue color and
	# define mask
	blue_lower = np.array([90, 150, 70], np.uint8)
	blue_upper = np.array([128, 255, 255], np.uint8)
	blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Set range for yellow color and
	# define mask
	yellow_lower = np.array([25, 80, 100], np.uint8)
	yellow_upper = np.array([40, 255, 255], np.uint8)
	yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
	
	# Morphological Transform, Dilation
	# for each color and bitwise_and operator
	# between imageFrame and mask determines
	# to detect only that particular color
	kernal = np.ones((5, 5), "uint8")
	
	# For red color
	red_mask = cv2.dilate(red_mask, kernal)
	res_red = cv2.bitwise_and(imageFrame, imageFrame,
							mask = red_mask)
	
	# For green color
	green_mask = cv2.dilate(green_mask, kernal)
	res_green = cv2.bitwise_and(imageFrame, imageFrame,
								mask = green_mask)
	
	# For blue color
	blue_mask = cv2.dilate(blue_mask, kernal)
	res_blue = cv2.bitwise_and(imageFrame, imageFrame,
							mask = blue_mask)

	# Creating contour to track white color
	contours, hierarchy = cv2.findContours(white_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(0, 0, 255), 2)
			
			cv2.putText(imageFrame, "White" + str(area), (x, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255))

	# Creating contour to track red color
	contours, hierarchy = cv2.findContours(red_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(0, 0, 255), 2)
			
			cv2.putText(imageFrame, "Red", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255))

	# Creating contour to track orange color
	contours, hierarchy = cv2.findContours(orange_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(0, 0, 255), 2)
			
			cv2.putText(imageFrame, "Orange", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255))		

	# Creating contour to track green color
	contours, hierarchy = cv2.findContours(green_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(0, 255, 0), 2)
			
			cv2.putText(imageFrame, "Green", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (0, 255, 0))

	# Creating contour to track blue color
	contours, hierarchy = cv2.findContours(blue_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(255, 0, 0), 2)
			
			cv2.putText(imageFrame, "Blue", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 0, 0))

	# Creating contour to track yellow color
	contours, hierarchy = cv2.findContours(yellow_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(5000 > area > 2000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(255, 0, 0), 2)
			
			cv2.putText(imageFrame, "Yellow", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 0, 0))   
			
	# Program Termination
	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
