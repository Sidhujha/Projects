from cvzone.FaceDetectionModule import FaceDetector
import ultralytics
import cv2
import cvzone
from time import time

###############################################

outputfolderpath="dataset/datacollect"
classID=1
offsetPercentagew=10
offsetPercentageH=20
confidence=0.8
save=True
debug=False
blurthreshold=35

###############################################

cap=cv2.VideoCapture(0)
detector=FaceDetector()

while True:
    success,img=cap.read()
    imgout=img.copy()
    img,bboxs=detector.findFaces (img,draw=False)
    
    listblur=[]
    listinfo=[]

    if bboxs:
        for bbox in bboxs:
            x, y, w, h =bbox["bbox"]
            score=bbox["score"][0]

            if score>confidence:
                offsetw=(offsetPercentagew / 100) * w
                x=int(x - offsetw)
                w=int(w + offsetw * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 2)
                h = int(h + offsetH * 2)

                if x<0:x=0
                if y<0:y=0
                if w<0:w=0
                if h<0:h=0

                imgface=img[y:y+h,x:x+w]
                cv2.imshow("face",imgface)
                blurvalue=int(cv2.Laplacian(imgface,cv2.CV_64F).var())

                if blurvalue>blurthreshold:
                    listblur.append(True)
                else:
                    listblur.append(False)


                ih,iw,_=img.shape
                xc,yc=x+w/2,y+h/2
                xcn,ycn=round(xc/iw,6),round(yc/ih,6)
                wn,hn=round(w/iw,6),round(h/ih,6)

                if xcn<1:xcn=1
                if ycn<1:ycn=1
                if wn<1:wn=1
                if hn<1:hn=1

                listinfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                cv2.rectangle(imgout, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgout,f'score:{int(score*100)}% blur: {blurvalue}',(x,y-20),scale=1,thickness=2)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img,f'score:{int(score*100)}% blur: {blurvalue}',(x,y-20),scale=1,thickness=2)
        
        if save:
            if all(listblur) and listblur!=[]:
                timenow=time()
                timenow=str(timenow).split('.')
                timenow=timenow[0]+timenow[1]
                cv2.imwrite(f"{outputfolderpath}/{timenow}.jpg",img)

                for info in listinfo:
                    f=open(f"{outputfolderpath}/{timenow}.txt",'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image",imgout) 
    cv2.waitKey(1)