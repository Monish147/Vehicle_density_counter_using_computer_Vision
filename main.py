import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist

# we make use of pretrained model for object detection 
model=YOLO('yolov8s.pt')



# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)
        

# cv2.namedWindow('ROI')
# cv2.setMouseCallback('ROI', RGB)

cap=cv2.VideoCapture('/home/monish/Desktop/digital_image_processing/mini_project/freedomweb/yolov8counting-trackingvehicles-main/nice_road_video.mp4')


my_file = open("/home/monish/Desktop/digital_image_processing/mini_project/freedomweb/yolov8counting-trackingvehicles-main/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

# counters for each class of our intrest 
car_counter = []
truck_counter = []
bike_counter = []

main_counter=[car_counter , truck_counter , bike_counter]

# object tracker 
tracker=Tracker()

# line y-coordinates for filtering the tracked objects 
cy1=140
cy2=240

# based on he speed of the vehicle we need to set the offset for efficient tracking 
offset_lst=[3,4,5]

# speed tracker list
# stores the time of the vehicle when it croses 1st line  
#vh_down = {}




def updater(category , counter , offset,roi):
    bbox_id=tracker.update(category)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        cv2.rectangle(roi,(x3,y3),(x4,y4),(0,0,255),2)


        if cy2<(cy+offset) and cy2 > (cy-offset):
            if counter.count(id)==0:
                counter.append(id)
                print(counter)
                cv2.circle(roi,(cx,cy),4,(0,0,255),-1)
                cv2.putText(roi,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        
        # if cy1<(cy+offset) and cy1 > (cy-offset):
        #     vh_down[id]=time.time()
        #     # print(vh_down)
        # if id in vh_down:
        #     if cy2<(cy+offset) and cy2 > (cy-offset):
        #         elapsed_time=(time.time() - vh_down[id])/36000
        #         # print(elapsed_time)
        #         if counter.count(id)==0:
        #             counter.append(id)
        #             distance = 1 # meters
        #             a_speed_ms = distance / elapsed_time
        #             a_speed_kh = a_speed_ms * 3.6
        #             cv2.circle(roi,(cx,cy),4,(0,0,255),-1)
        #             cv2.putText(roi,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
        #             cv2.putText(roi,str(int(a_speed_kh))+'Km/h',(x3,y3 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    return


while True:
    ret,frame = cap.read()

    roi = frame[400:1000 ,700:1300]

    if not ret:
        break
    # count += 1
    # if count % 3 != 0:
    #     continue
    frame=cv2.resize(frame,(1020,500))

    # main_list is used to store all the sub list of each category (car , truck , bike)
    main_list=[]
    # car sub list to store coordinates of the detected cars
    car_list = []
    # truck sub list to store coordinates of the detected trucks
    truk_list = []
    # bike sub list to store coordinates of the detected bike
    bike_list = []

    results=model.predict(roi)
 
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
             
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c :
            car_list.append([x1,y1,x2,y2])
        elif 'truck' in c or 'bus' in c:
            truk_list.append([x1,y1,x2,y2])
        elif 'motorcycle' in c or 'bicycle' in c:
            bike_list.append([x1,y1,x2,y2])

    # add all these sub list to a main list and so that we can track each vehicle category in a loop 
    main_list = [car_list,truk_list,bike_list]


    for i in range(len(main_list)):
        updater(main_list[i] , main_counter[i] , offset_lst[i] , roi)
        


    cv2.line(roi,(50,cy1),(450,cy1),(255,255,255),1)
    cv2.line(roi,(1,cy2),(530,cy2),(255,255,255),1)
    cv2.putText(roi , ('car_count - ')+str(len(car_counter)) , (10,50) , cv2.FONT_HERSHEY_COMPLEX,0.8 , (0,255,0),2)
    cv2.putText(roi , ('truck_count - ')+str(len(truck_counter)) , (10,25) , cv2.FONT_HERSHEY_COMPLEX,0.8 , (0,0,255),2)
    cv2.putText(roi , ('bike_count - ')+str(len(bike_counter)) , (10,75) , cv2.FONT_HERSHEY_COMPLEX,0.8 , (255,0,0),2)
    cv2.imshow("RGB", frame)
    cv2.imshow("ROI", roi)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()