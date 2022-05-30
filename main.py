import cv2
import imutils
import numpy as np
import detech_color as dc

color_mask = []
    
img = cv2.imread('./rubik_1.png')
# 48
# 49
# 66
img2 = img.copy()   
cv2.imshow('Input', img)

# edge = cv2.Canny(img3, 100, 200, L2gradient = True)
# cv2.imshow('edge', edge)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow('Gray of rubik', gray)

# Apply threshold

ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
# thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,10)
cv2.imshow('After apply threshold', thresh)

cnts, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cnt_max = []
mx_area = -1

cv2.drawContours(img, cnts, -1, (0,255,0), 3)
cv2.imshow('Contour of rubik', img)

# Track object rubik

for i in cnts:
    if mx_area <= cv2.contourArea(i):
        mx_area = cv2.contourArea(i)
        cnt_max = i
rect = cv2.minAreaRect(cnt_max) # Tìm thông tin của hình chữ nhật nhỏ nhất
box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(img2, [box], 0, (36, 255, 12), 3)
cv2.drawContours(img, [box], 0, (36, 255, 12), 3)

cv2.imshow('Contour rectangle of rubik', img2)

width = int(rect[1][0]) # get width
height = int(rect[1][1]) # get height
src_pts = box.astype("float32")
dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img2, M, (width, height))  # will be used further
cv2.imshow('Rubik after tracking', warped)

warped = cv2.resize(warped, (480, 480))
warped[:,0:8,:] = 0
warped[:,warped.shape[0]-9:,:] = 0
warped[0:8,:,:] = 0
warped[warped.shape[1]-8:,:,:] = 0

if rect[2] >= 45:
    M = cv2.getRotationMatrix2D((240, 240), -90, 1.0)
    rotated = cv2.warpAffine(warped, M, (480, 480))
elif rect[2] <= -45:
    M = cv2.getRotationMatrix2D((240, 240), -90, 1.0)
    rotated = cv2.warpAffine(warped, M, (480, 480))
else :
    rotated = warped.copy()

img3 = rotated.copy()
hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)


kernal = np.ones((5,5), "uint8")
res = []
color_mask = []
lower1 = []
upper1 = []
# Get mask of color 

for i in dc.color.values():
    hsv_lower = np.array(dc.FLOOR_THRESHOLD[i], dtype=np.uint8)
    hsv_upper = np.array(dc.CEIL_THRESHOLD[i], dtype=np.uint8)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.dilate(mask,kernal)
    res.append(cv2.bitwise_and(img3, img3, mask = mask))
    color_mask.append(mask)


contours = []

# Find contour of color mask

for i in range(6):
    (c, h) = cv2.findContours(color_mask[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours.append(c)

# Track color depend on contour

idx = 0
a = np.zeros((9,3), dtype = 'int')
tmp = 0

for i in dc.color.keys():
    for contour in contours[i - 1]:
        area = cv2.contourArea(contour)
        if(area > 3000):
            x,y,w,h = cv2.boundingRect(contour)
            # a[idx] = (x, y, i)
            # idx += 1
            img3 = cv2.rectangle(img3, (x,y), (x+w,y+h), dc.BGR_COLOR[dc.color[i]], 2)
            img3 = cv2.line(img3,(x,y),(x+w,y+h),dc.BGR_COLOR[dc.color[i]],2)
            cv2.putText(img3,dc.color[i], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dc.BGR_COLOR[dc.color[i]])

# print(a)
# a = sorted(a, key= lambda y_Index: y_Index[1] , reverse = False)
# for i in range(0,3):
#     a[i*3:i*3+3] = sorted(a[i*3:i*3+3], key= lambda x_Index: x_Index[0] , reverse = False)

# color_detect = []
# for i in range(0,9):
#     color_detect.append(dc.color[a[i][2]])

# print("Color of face rubik :\n ",  np.array(color_detect).reshape((3, 3)))


cv2.imshow("Ouput",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()