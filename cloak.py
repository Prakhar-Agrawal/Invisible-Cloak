import numpy as np
import cv2
import time
#import pytesseract 
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


video = cv2.VideoCapture(0)
time.sleep(3)

background = 0

#capturing the background here
for i in range(30):
    ret, background = video.read() #here we are capturing the background only, with this video.read() 

#this video.isOpened() function is used to run the while loop only till the webcam is capturing our video. 
#as soon as we close the window, the loops stops running
#if we used while(true) then it would run even if we close the webcam video window.

while(video.isOpened()):
    ret, img = video.read() #But here we are capturing video through webcam to perform operation on it
    if not ret:
        break
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #HSV = Hue Saturation Value #OR HSB = Hue Saturation Brightness

    lower_red = np.array([0,120,65])#these are h,s,v values
    upper_red = np.array([9,255,255])#these are h,s,v values
    mask1 = cv2.inRange(hsv, lower_red, upper_red) #separating the cloak part

    lower_red = np.array([170,120,70])#these are h,s,v values
    upper_red = np.array([180,255,255])#these are h,s,v values
    mask2 = cv2.inRange(hsv, lower_red, upper_red)#separating the cloak part

    mask1 = mask1 + mask2 # this + is bitwise 'or' i.e. if there is any shade of red in range either 
                            # 0-10 or 170-180 then it would be segmented and stored in mask1

    mask1 = cv2.erode(mask1,np.ones((5,5),np.uint8),iterations=2)

    mask1 = cv2.dilate(mask1,np.ones((5,5),np.uint8),iterations=1)
    #mask3=mask1
    #text = pytesseract.image_to_string(mask3, lang='eng')
    #print(text)


    mask2 =cv2.bitwise_not(mask1) #Everything except mask1 i.e. other than cloak exerything would be there.
    res1 = cv2.bitwise_and(background,background,mask=mask1) #used for segmentation of the color 
    res2 = cv2.bitwise_and(img,img,mask=mask2) #used to substitute the cloak part
    final_output = cv2.addWeighted(res1,1,res2,1,0)

    cv2.imshow('Eureka!!', final_output)
    k = cv2.waitKey(10)
    if k==27:
        break
video.release()


print("video released")
cv2.destroyAllWindows()
