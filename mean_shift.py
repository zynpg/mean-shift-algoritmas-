import cv2
import numpy as np

cap=cv2.VideoCapture(0)

ret, frame= cap.read()
if not ret:
    print("kamera açılamadı")
    exit()

#yüz algılama için Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(frame)



#listenin boş olup olmadığına bakılır
if len(faces)==0:
    print("yüz bulunamadı")
    exit()

(x, y, w, h)=faces[0]
track_window= (x,y,w,h) #takip penceresi

#region of insterest ilgi bölgesi
roi= frame[y:y+h, x:x+w]


hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.,)), np.array((180., 255. , 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


#meanshift kriterlerini ayarla
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        dist= cv2.calcBackProject([hsv],[0], roi_hist,[0,180],1)

        ret,track_window= cv2.meanShift(dist,track_window, term_crit)

        x,y,w,h= track_window

        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        cv2.imshow("Yüz",img)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()



















