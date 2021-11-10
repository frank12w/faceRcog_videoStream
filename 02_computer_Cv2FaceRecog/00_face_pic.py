import cv2
import urllib.request
import numpy as np
 
#f_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
f_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
##f_cas= cv2.CascadeClassifier('/home/wiser/21_esp32_streamingVideo/computer_Cv2FaceRecog/haarcascade_frontalface_default.xmL')
##eye_cascade=cv2.CascadeClassifier('/home/wiser/21_esp32_streamingVideo/computer_Cv2FaceRecog/haarcascade_eye.xml')
url='http://192.168.0.140/cam-lo.jpg'
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
cv2.namedWindow("Live Transmission", cv2.WINDOW_AUTOSIZE)

i = 0
j = 1
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    img2=cv2.imdecode(imgnp,-1) #for saving ori image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=f_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            i += 1
            if i%5==0:
                imgname = 'faceImg_' + str(j) + '.jpg'
                cv2.imwrite(imgname, img2)
                print('save %s' %imgname)
                j += 1
            

 
 
    cv2.imshow("live transmission",img)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
 
cv2.destroyAllWindows()