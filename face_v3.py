#Import required modules
import cv2
import dlib
import scipy.misc
import numpy as np
import pyautogui
from win32api import GetSystemMetrics
cursor_x_axis=cursor_y_axis=0
num_seconds=.0001
#Set up some requirqed objects
face_cascade = cv2.CascadeClassifier('xmlfile/haarcascade_frontalface_alt.xml')
#video_capture = cv2.VideoCapture("http://192.168.43.1:8080/video") #Webcam object
#video_capture=cv2.VideoCapture("http://192.168.43.1:8080/video")
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
nosetip=30
lelc=36
leup=38
ledn=41
lerc=39
rerc=45
reup=44
redn=47
relc=42
mlc=48
mrc=54
mup=51
mdn=66
left_eyebrow=19
right_eyebrow=24
cx=0
cy=0
flag=0
while True:
    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1)
    print frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        gray=gray[y:y+h,x:x+w]
        newf=frame[y:y+h,x:x+w]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1) #Detect the faces in the image

        for k,d in enumerate(detections): #For each detected face

            shape = predictor(clahe_image, d) #Get coordinates
            
            cv2.circle(newf, (shape.part(8).x, shape.part(8).y), 1, (0,0,255), thickness=1)
            # left eye corners
            #cv2.circle(newf, (shape.part(lelc).x, shape.part(lelc).y), 1, (0,0,255), thickness=1)
            #cv2.circle(newf, (shape.part(lerc).x, shape.part(lerc).y), 1, (0,0,255), thickness=1)
            # mouth
            cv2.circle(newf, (shape.part(mrc).x, shape.part(mrc).y), 1, (0,0,255), thickness=1)
            cv2.circle(newf, (shape.part(mlc).x, shape.part(mlc).y), 1, (0,0,255), thickness=1)
            #nose
            cv2.circle(newf, (shape.part(nosetip).x, shape.part(nosetip).y), 1, (0,0,255), thickness=1)
            cv2.circle(newf, (shape.part(left_eyebrow).x, shape.part(left_eyebrow).y), 1, (0,0,255), thickness=1)
            # right eye corners
            #cv2.circle(newf, (shape.part(rerc).x, shape.part(rerc).y), 1, (0,0,255), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
            #cv2.circle(newf, (shape.part(relc).x, shape.part(relc).y), 1, (0,0,255), thickness=1)
            #------ ----------making right eye configuration------------- --------------#
            leftcorn_x_axis=shape.part(relc).x+x # +x is done for absolute value on screen
            leftcorn_y_axis=shape.part(relc).y+y # +y is done for absolute value on screen
            rigtcorn_x_axis=shape.part(rerc).x+x # +x is done for absolute value on screen
            rigtcorn_y_axis=shape.part(rerc).y+y # +y is done for absolute value on screen 
            rigtup_x_axis=shape.part(reup).x+x   # +x is done for absolute value on screen
            rigtup_y_axis=shape.part(reup).y+y
            rigtdn_x_axis=shape.part(redn).x+x
            rigtdn_y_axis=shape.part(redn).y+y
            
            #-------- making left eye configuration --------------------------#
            lefleftcorn_x_axis=shape.part(lelc).x+x # +x is done for absolute value on screen
            leflefcorn_y_axis=shape.part(lelc).y+y # +y is done for absolute value on screen
            lefrigtcorn_x_axis=shape.part(lerc).x+x # +x is done for absolute value on screen
            rigtcorn_y_axis=shape.part(lerc).y+y # +y is done for absolute value on screen 
            leftup_x_axis=shape.part(leup).x+x   # +x is done for absolute value on screen
            leftup_y_axis=shape.part(leup).y+y
            leftdn_x_axis=shape.part(ledn).x+x
            leftdn_y_axis=shape.part(ledn).y+y
            #iris_x_axis=cx+shape.part(lelc).x+x+2  # absolute x-axis for iris
            #iris_y_axis=cy+shape.part(leup).y+y+3 # absolte y-axis for iris
            #cv2.rectangle(frame,(leftcorn_x_axis,leftup_y_axis),(rigtcorn_x_axis,leftdn_y_axis),(255,0,0,),2)            
           
            # extracting the right eye
            roi=newf[shape.part(reup).y-3:shape.part(redn).y+3,shape.part(relc).x+2:shape.part(rerc).x-2]
            resize=cv2.resize(roi, (0,0), fx=1, fy=1)
            # extracting the left eye
            lefteye=frame[leftup_y_axis:leftdn_y_axis,lefleftcorn_x_axis:lefrigtcorn_x_axis]
            # converting to gray scale image
            g_resize=cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
            left_g_resize=cv2.cvtColor(lefteye,cv2.COLOR_BGR2GRAY)
            # changing the contrast of the right eye 
            equ = cv2.equalizeHist(g_resize)
            left_equ=cv2.equalizeHist(left_g_resize)
            # extracting the eye ball based on color range
            thres=cv2.inRange(equ,0,15)
            left_thres=cv2.inRange(left_equ,0,15)
            # creating a kernel of 3x3
            kernel = np.ones((5,5),np.uint8)
            #/------- removing small noise inside the white image ---------/#
            dilation = cv2.dilate(thres,kernel,iterations = 4)
            left_dilation = cv2.dilate(left_thres,kernel,iterations = 4)
            #/------- decreasing the size of the white region -------------/#
            erosion = cv2.erode(dilation,kernel,iterations = 2)            
            left_erosion = cv2.erode(left_dilation,kernel,iterations = 2) 
            cv2.imshow("roi",left_erosion)
            # creating the countor around the eyeball
            image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            limage, lcontours, lhierarchy = cv2.findContours(left_erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            if len(contours)==2 :
            #numerator+=1
            #img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            #------ finding the centroid of the contour ----------------#
                M = cv2.moments(contours[0])
                #print M['m00']
                #print M['m10']
                #print M['m01']
                if M['m00']!=0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.line(roi,(cx,cy),(cx,cy),(0,0,255),3)
            elif len(contours)==1:
                #numerator+=1
                #img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

                #------- finding centroid of the countor ----#
                M = cv2.moments(contours[0])
                if M['m00']!=0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    #print cx,cy
                    cv2.line(roi,(cx,cy),(cx,cy),(0,0,255),3)
    #--------countor for left eye ------------------------------------------#
            if len(lcontours)==2 :
            #numerator+=1
            #img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            #------ finding the centroid of the contour ----------------#
                M = cv2.moments(lcontours[0])
                #print M['m00']
                #print M['m10']
                #print M['m01']
                if M['m00']!=0:
                    lcx = int(M['m10']/M['m00'])
                    lcy = int(M['m01']/M['m00'])
                    cv2.line(lefteye,(lcx,lcy),(lcx,lcy),(0,0,255),3)
            elif len(contours)==1:
                #numerator+=1
                #img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

                #------- finding centroid of the countor ----#
                M = cv2.moments(lcontours[0])
                if M['m00']!=0:
                    lcx = int(M['m10']/M['m00'])
                    lcy = int(M['m01']/M['m00'])
                    #print cx,cy
                    cv2.line(lefteye,(lcx,lcy),(lcx,lcy),(0,0,255),3)
            #------ window resolution-----------#
            screen_width=GetSystemMetrics(0)
            screen_height=GetSystemMetrics(1)

            iris_x_axis=cx+shape.part(relc).x+x  # absolute x-axis for iris
            iris_y_axis=cy+shape.part(reup).y+y # absolte y-axis for iris
            left_iris_x_axis=cx+shape.part(lelc).x+x  # absolute x-axis for iris
            left_iris_y_axis=cy+shape.part(leup).y+y # absolte y-axis for iris

            lshift=(cx-17)*30
            upshift=(cy-5)*50
            #print upshift
            left_lshift=(lcx-10)*30
            left_upshift=(lcy-5)*50

            cursor_x_axis=iris_x_axis+lshift
            cursor_y_axis=iris_y_axis+upshift
            left_cursor_x_axis=left_iris_x_axis+lshift
            left_cursor_y_axis=left_iris_y_axis+upshift
            avg_cursor_x_axis=(cursor_x_axis+left_cursor_x_axis)/2
            avg_cursor_y_axis=(cursor_y_axis+left_cursor_y_axis)/2
            #print cursor_x_axis,cursor_y_axis,left_cursor_x_axis,left_cursor_y_axis
             
            moup_to_nose=(shape.part(mup).y-shape.part(nosetip).y)
            modn_to_nose=(shape.part(mdn).y-shape.part(nosetip).y)
            cur_cont=float(modn_to_nose)/float(moup_to_nose)
            print cur_cont
            #------ loking the cursor----------------------#
            if cur_cont>=1.4:
                if flag==0:
                    flag=1
                    print "cursor off"
                else:
                    flag=0
                    print "cursor on"

            #-------- cursor position detection ------------------------------#
            cv2.line(frame,(left_iris_x_axis,left_iris_y_axis),(avg_cursor_x_axis,avg_cursor_y_axis),(255,0,0),3)
            cv2.line(frame,(iris_x_axis,iris_y_axis),(avg_cursor_x_axis,avg_cursor_y_axis),(255,0,0),3)
           
            #---------- checking for eye blink ------------------#
            eyeblink=shape.part(redn).y-shape.part(reup).y
            left_eyeblink=shape.part(ledn).y-shape.part(leup).y
            #------- left click ----------------------------#
            if eyeblink<=2:
                pyautogui.click()
            #------- right click ---------------------------#
            if left_eyeblink<=4:
                pyautogui.click(button='right')

    cv2.imshow("image", frame) #Display the frame
    if flag==0:
        pyautogui.moveTo(avg_cursor_x_axis,avg_cursor_y_axis, duration=num_seconds)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break