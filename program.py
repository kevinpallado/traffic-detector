from datetime import date, datetime , timedelta
import math
import mysql.connector
import cv2
import sys
import math
import time as t
import csv
import codecs
import numpy as np
import scipy
# import math

# video_source = "tra2.avi"
#video_source = "car3.mp4"
video_source = "Vid_ID6_evening.avi"
# video_source = "BritishHighwayTraffic.mp4"
# video_source = "LondonHeathrowTraffic.webm"
# video_source = "ManhattanTraffic.mp4"
# video_source = "Alibi.avi"
counter = 0
dp_count = 0
long_vehicle = 0
total_cars = 0
length_cars = 0
total_scars = 0
short_vehicle = 0
total_long = total_short = clong_vehicle = cshort_vehicle =0
car_index_count = [None]*4
# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
TrackerType = "MEDIANFLOW"
fgbg = cv2.createBackgroundSubtractorMOG2()
# TrackerType = "TLD"
# TrackerType = "KCF"
btch = 0
save_id = 0
cascadePath = "car4-1.xml"
start = t.time()
video_id = 6
prev_car_index = None
car_saved = 0
cnnct = mysql.connector.connect(user='root', password='',
                          host='127.0.0.1',
                          database='vehicle_count')
cursor = cnnct.cursor()
today = datetime.today()
add_data = ("INSERT INTO traffic_counting "
            "(video_id,class2_vehicle, class1_vehicle,total_cars,date_stamp,time_added)"
            "VALUES (%s, %s, %s, %s, %s,%s)")


def cent_dist(a,b):
    temp = math.sqrt( pow((a[1]-b[1]),2)+pow((a[0]-b[0]),2))
    return temp

def checkOverlap(a,b):  
    (x1, y1, w1, h1) = a
    (x2, y2, w2, h2) = b

    if(x1 < (x2 +w2/2)):
        if(x1+w1 > (x2 +w2/2)):
            if(y1 < (y2 +h2/2)):
                if(y1+h1 > (y2 +h2/2)):
                    return True
    if(x2 < (x1 +w1/2)):
        if(x2+w2 > (x1 +w1/2)):
            if(y2 < (y1 +h1/2)):
                if(y2+h2 > (y1 +h1/2)):
                    return True
    return False

def saveDatabase(carCount,long_v,short_v,currentTime):
    data_add = (video_source,long_v,short_v,carCount,today,currentTime)
    #data_add = (4,long_v,short_v,carCount,today,currentTime)
    cursor.execute(add_data,data_add)
    cnnct.commit()
    
def removeOverlaps(objectsFoundLocal):
    objectsFoundTemp = []
    for i in range(0, len(objectsFoundLocal)):
        matchBool = False
        for j in range(i+1, len(objectsFoundLocal)):
            if (checkOverlap(objectsFoundLocal[i],objectsFoundLocal[j])):
                matchBool = True
        if not matchBool:
            objectsFoundTemp.append(objectsFoundLocal[i])
    return objectsFoundTemp
tracker          = {}
status           = {}
trackerLifeTime  = {}
bbox             = {}
bboxOld          = {}
ok               = {}
centroid_car     = {}
centroid_tracker = {}
Dir              = {}
activeTrackers = np.array(0)
no_trackers=35

def deactivateTracker(indexOfTracker):
    global activeTrackers
    index = np.argwhere(activeTrackers==indexOfTracker)
    # print ("index = ", index)
    activeTrackers = np.delete(activeTrackers, index)

for i in range(0,no_trackers):
    trackerLifeTime[i] = 0
    tracker[i]=cv2.TrackerMedianFlow_create()
    status[i] = "OFF"
    ok[i]=False

#cascade init
car_cascade = cv2.CascadeClassifier(cascadePath)

# Video Initialize
video = cv2.VideoCapture(video_source)
width = video.get(3)
height = video.get(4)
print ("height : %d width : %d ")%(height,width)
height_int = int(height)
width_int = int(width)
# Exit if video not opened.
if not video.isOpened():
    print ("Could not open video")
    sys.exit()

# Read first frame.
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
# frame = frame[:,(frame.shape[1]/2):frame.shape[1]].copy()
frameTrackersPrev = frame.copy()
frameHaarPrev     = frame.copy()
if not ok1:
    #print ('Cannot read video file')
    sys.exit()

#convt gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detect cars
cars = car_cascade.detectMultiScale(gray, 1.3, 5)


#Remove overlaping cars
cars = removeOverlaps(cars)
# cars = esvm_nms(cars, 30)

#initialize all tracker related arrays
for i in range(0, no_trackers):
    bbox[i]    = (0, 0, 0, 0)
    bboxOld[i] = (0, 0, 0, 0)
    tracker[i] = cv2.TrackerMedianFlow_create()
    ok[i]      = tracker[i].init(frame, bbox[i])
    tracker[i] = cv2.TrackerMedianFlow_create()
    status[i]  = "OFF"
    Dir[i]     = "IN"
    ok[i]      = False

trackersOn     = 0  #counter for No of active trackers
#carCount       = 0     #car counter
#carCountIn     = 0     #car counter
#carCountOut    = 0     #car counter
pause          = False
totT1 = 0
totT15 = 0
totT2 = 0
totT3 = 0
totT4 = 0
totT5 = 0
totT6 = 0
cars_count = 0
traffic_cdo = 0
totalFrames    = 0
stTime         = t.time()
absoluteStTime = stTime
while True:
    ui_data = np.zeros((640,360,3), np.uint8)
    cv2.putText(ui_data,'Video ID :',(10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,str(video_source),(120,50),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),1)
    cv2.putText(ui_data,'Total Vehicle(s) : ',(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Class 2 Vehicles : ',(10,150),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Class 1 Vehicles : ',(10,200),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Car Deteced',(120,250),cv2.FONT_HERSHEY_SIMPLEX, 0.50,(255, 255, 255),1)
    cv2.putText(ui_data,'Car Index : ',(10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Car Length(PX) : ',(10,350),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Database Records',(100,400),cv2.FONT_HERSHEY_SIMPLEX, 0.50,(255, 255, 255),1)
    cv2.putText(ui_data,'Total Vehicle(s) : ',(10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Class 2 Vehicles : ',(10,500),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Class 1 Vehicles : ',(10,550),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    cv2.putText(ui_data,'Time Recorded : ',(10,600),cv2.FONT_HERSHEY_SIMPLEX, 0.60,(255, 255, 255),2)
    activeTrackers = np.unique(activeTrackers)
    # Read a new frame
    # print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    # print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    # print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    # print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    # print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    # print("")
    # print("")
    ok1, frame = video.read()
    #print(frame.shape)
    if not ok1:
        break
    frame = cv2.resize(frame, (360,640))
    #print(frame.shape)
    #frame = cv2.transpose(frame)
    #frame = frame[:450,:]

    # frame = frame[:,:300]
    # frame = frame[:,(frame.shape[1]/2):frame.shape[1]].copy()
    frameHaar = frame.copy()
    frameTrackers = frame.copy()
    fgmask = fgbg.apply(frameTrackers)
    #convt gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time1 = t.time()
    T1 = time1 - stTime
    #print("\nTime1 = ", T1)

    #detect cars
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)

    time15 = t.time()
    T15 = time15 - time1
    #print("Time15 = ", T15)
    #Remove overlaping cars
    cars = removeOverlaps(cars)
    # print("len(cars): ", len(cars))
    #cv2.line(frameTrackers,(370,300),(520,300),(255,0,0),5) #iphone
    #cv2.line(frameTrackers,(240,420),(260,420),(255,0,0),5) for recorded video
    cv2.line(frameTrackers,(120,330),(250,330),(255,0,0),5)
    cv2.line(frameTrackers,(130,330),(110,380),(255,255,255),3)
    cv2.line(frameTrackers,(245,330),(270,380),(255,255,255),3)
    cv2.line(frameTrackers,(110,380),(270,380),(255,255,255),3)
    width4 = width_int/4
    height2 = height_int/2
    #print height2
    #print width4
    #cv2.line(frameTrackers,(200,height_int/2),(460,height_int/2),(255,0,0),5)
    #prepare centroid of Haar cars
    i=0
    for (x,y,w,h) in cars:
        centroid_car[i]=(x+(w/2),y+(h/2))
        i+=1
        #if x >= 120 and x <= 250 and y == 420:
            #roi = frame[y:y+h, x:x+w]
            #cv2.rectangle(roi,(x,y),(x+w,y+h), (0, 255, 0), 2)
            #cars_count += 1
            #dp_count = dp_count + cars_count
            #print ("x = %d y = %d")%(x,y)
            #print ("w = %d h = %d")%(w,h)
            #print ("current cars = %d")%cars_count
            #print ("total cars = %d")%dp_count
            
            #roi_g = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            #cv2.imshow('roi',roi_g)
            #counter +=1
            #print type(roi)
            #cv2.imwrite('roi-images' + str(counter) + '.png', roi_g)
            #print counter
            #print cv2.contourArea(roi)
            #tup = tuple(map(tuple, roi))
            #imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #thresh = cv2.threshold(imgray, 127, 255,0)
            #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(roi, contours, -1, (0,255,0), 3)
            #roi  = fgbg.apply(imgray)
            
            #dp_count = dp_count1 + cars_count

    #prepare centroid of Tracker cars   (Not used for Computation. Used only for printing and debugging)
    for i in range(0,no_trackers):
        if status[i] != "OFF":
            temp = (bbox[i][0]+(bbox[i][2]/2),bbox[i][1]+(bbox[i][3]/2))
            # centroid_tracker[i]= temp
        else:
            centroid_tracker[i]=(0,0)

    # print ("car: ",     centroid_car)
    # print ("tracker: ", bbox,"\n\n")
    # print ("tracker: ", centroid_tracker,"\n\n")

    time2 = t.time()
    T2 = time2 - time15
    #print("Time2 = ", T2)
    for i in range(0,len(cars)):
        matchFound = False
        # for j in range (0,no_trackers):     #checck if any trackers are already tracking the haar car
        for j in activeTrackers:
            if status[j] == "DUP":
                continue
            # if status[j] == "OFF":
            #     continue
            # if status[j] == "LOST":
            #     print ( "klmvkvs")
            #     bbox[j] = bboxOld[j]
            #     print ("bbox = ", bbox)
            #     print ("bboxOld = ", bboxOld)
            if ((checkOverlap(cars[i],bbox[j])) | (checkOverlap(bbox[j],cars[i]))):
                    # print ("Overlap >> CAR: ", cars[i], "TRACKER: ", bbox[j], end = '  ')
                    # print ("      >> CAR: ", i, "TRACKER: ", j)
                    if((cars[i][2] < bbox[j][2]) & (matchFound == False)):
                    # if((matchFound == False)):
                        p1 = (int(bbox[j][0]), int(bbox[j][1]))
                        p2 = (int(bbox[j][0] + bbox[j][2]), int(bbox[j][1] + bbox[j][3]))
                        # cv2.putText(frameTrackers,(str)(j),(int(bbox[j][0] + 5) ,int(bbox[j][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255),2)
                        # cv2.putText(frameTrackers,(str)(status[j][0]),(int(bbox[j][0] + 5) ,int(bbox[j][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255),2)
                        # print ( "FTrack Text: car: ", i,"tr: ", j, " Col: Red")
                        # cv2.rectangle(frameTrackers, p1, p2, (0,0,255), 1)
                        if status[j] == "NEW":
                            continue
                        #     status[j] = "UD"
                        tracker[j]=cv2.TrackerMedianFlow_create()
                        temp = (cars[i][0], cars[i][1], cars[i][2], cars[i][3])
                        bbox[j] = temp
                        ok[j] = tracker[j].init(frame, temp)
                        # if status[j] == "LOST":
                        #     # status[i] = "LOST"
                        #     # print ("trackerOn -= 1", i, "st = ", status[i])
                        #     trackersOn += 1
                        #     pause = True
                        #     carCount -= 1
                        #     if(Dir[j] == "IN"):
                        #         carCountIn  -= 1
                        #     elif(Dir[j] == "OUT"):
                        #         carCountOut -= 1
                        # print ("Tracker Updated TrNo: ", j, "CarNo: ", i)
                        status[j] = "HUD"
                        matchFound = True
                        p1 = (int(bbox[j][0]), int(bbox[j][1]))
                        p2 = (int(bbox[j][0] + bbox[j][2]), int(bbox[j][1] + bbox[j][3]))
                        # cv2.putText(frameTrackers,(str)(j),(int(bbox[j][0] + 5) ,int(bbox[j][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 0, 0),2)
                        # cv2.putText(frameTrackers,(str)(status[j][0]),(int(bbox[j][0] + 5) ,int(bbox[j][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 0, 0),2)
                        # print ( "FTrack Text: car: ", i,"tr: ", j, " Col: Blue")
                        # cv2.rectangle(frameTrackers, p1, p2, (255,0,0), 1)
                        # cv2.imshow('Haar', frameHaar)
                        # cv2.imshow('Trackers', frameTrackers)
                        # pause = True
                        # continue
                    # cv2.putText  (frameHaar, (str)(i),(cars[i][0] + 5 , cars[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # print ( "FHaar Text: car: ", i,"tr: ", j, " Col: Green")
                    cv2.rectangle(frameHaar, (cars[i][0], cars[i][1]), (cars[i][0]+cars[i][2], cars[i][1]+cars[i][3]), (0, 255, 0), 2)
                    matchFound = True
        if matchFound == False:             #if not already tracked, create new tracker
            for k in range(0,no_trackers):
                if status[k] == "OFF":               #check if tracker is available
                    # print ("Init New  >> CAR: ", cars[i], end = '  ')
                    # print ("NEW >> CAR: ", i, "TRACKER: ", k)
                    # for j in range (0,no_trackers):
                    #     print("car ",cars[i],": trck ",bbox[j]," = ",checkOverlap(cars[i],bbox[j]))
                    # cv2.putText(frameHaar,(str)(i),(cars[i][0] + 5 ,cars[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2)
                    # print ( "FHaar Text: car: ", i,"tr: ", j, " Col: Green")
                    # cv2.rectangle(frameHaar, (cars[i][0], cars[i][1]), (cars[i][0]+cars[i][2], cars[i][1]+cars[i][3]), (0, 0, 255), 2)
                    trackerLifeTime[k] = 0
                    tracker[k]=cv2.TrackerMedianFlow_create()
                    temp = (cars[i][0], cars[i][1], cars[i][2], cars[i][3])
                    bbox[k] = temp
                    ok[k] = tracker[k].init(frame, temp)
                    temp = (cars[i][0], cars[i][1], cars[i][2], cars[i][3])
                    bbox[k] = temp
                    trackersOn += 1
                    status[k] = "NEW"
                    activeTrackers = np.append(activeTrackers, k)
                    break
    # print("\n\n")

    time3 = t.time()
    T3 = time3 - time2
    #print("Time3 = ", T3)
    # Update tracker
    # print ("st = ", status)
    # for i in range(0,no_trackers):
    for i in activeTrackers:
        # if   status[i] == "OFF":      #If tracker is OFF do not update
        #     # print (i, ": OFF ", end='. ')
        #     pass
        if status[i] == "UD":      #If tracker is OFF do not update
            bboxOld[i] = bbox[i]
            # print (i, ": UPD", end='. ')
            if((bbox[i][0] + bbox[i][2]/2) < (frame.shape[1]/2)):
                Dir[i] = "OUT"
            else:
                Dir[i] = "IN"
            # tex1 = t.time()
            ok[i], bbox[i] = tracker[i].update(frame)
            # tex2 = t.time()
            # TEX = tex2 - tex1
            # print("TEX = ", TEX)
            trackerLifeTime[i] += 1
            #if not ok[i]:
                #status[i] = "LOST"
                # print ("trackerOn -= 1", i, "st = ", status[i])
                #trackersOn -= 1
                #pause = True
                #carCount += 1
                #if(Dir[i] == "IN"):
                #    carCountIn  += 1
                #elif(Dir[i] == "OUT"):
                #    carCountOut += 1
            #elif bbox[i][2]<20:     #Out Of Sight (Too Small)
                #status[i] = "LOST"
                # print ("trackerOn -= 1", i, "st = ", status[i])
                #trackersOn -= 1
                #carCount += 1
                #if(Dir[i] == "IN"):
                #    carCountIn  += 1
                #elif(Dir[i] == "OUT"):
                #    carCountOut += 1
                #pause = True
                #ok[i]=False
        elif status[i] == "NEW":      #If tracker is NEw do not update
            # print (i, ": NEW", end='. ')
            status[i] = "UD"
        elif status[i] == "HUD":      #If tracker is recently updated from Haar do not update
            # print (i, ": NEW", end='. ')
            status[i] = "UD"
        elif status[i] == "DUP":      #If tracker is duplicate do not update
            # print (i, ": DUP ", end='. ')
            status[i] = "OFF"
            bbox[i] == (0,0,0,0)
            ok[i]=False
        elif status[i] == "LOST":      #If tracker is LOST do not update
            # print (i, ": LOST ", end='. ')
            status[i] = "OFF"
            deactivateTracker(i)
            bbox[i] == (0,0,0,0)
            ok[i]=False

    time4 = t.time()
    T4 = time4 - time3
    #print("Time4 = ", T4)
    # print ("")
    # print ("ok = ",ok)
                
    # Remove Duplicate Trackers
    # trackersTemp = {}
    for i in range(0, len(bbox)):
        if(bbox[i] == (0,0,0,0)):
            continue
        if(status[i] == "OFF"):
            continue
        matchBool = False
        for j in range(i+1, len(bbox)):
            if(bbox[j] == (0,0,0,0)):
                continue
            if(status[j] == "OFF"):
                continue
            if (checkOverlap(bbox[i],bbox[j])):
                # print ("Dupl Tracker >> Track1: ", bbox[i], "Track2: ", bbox[j], end = '  ')
                # print ("      >> Tr1: ", i, "Tr2: ", j)
                # print ("initial Status: (i)", i, ": ", status[i], " (j)", j, ": ", status[j])
                status[i] = "DUP"
                trackersOn -= 1
                bbox[i] == (0,0,0,1)
                matchBool = True
    time5 = t.time()
    T5 = time5 - time4
    #print("Time5 = ", T5)

    # for i in range(0,no_trackers):
    for i in activeTrackers:
        if ok[i]:
            p1 = (int(bbox[i][0]), int(bbox[i][1]))
            p2 = (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3]))
            #object_2(p1,p2)
            if((status[i] == "UD")):

                #cv2.putText(frameTrackers,(str)(status[i][0]),(int(bbox[i][0] + 5) ,int(bbox[i][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 0),2)
                # print ( "FTrack Text: car: ", i,"tr: ", j, " Col: Green")
                #print "car index : %s" %(str)(i)
                #print "car y : %d" %int(bbox[i][1])
                # print int(bbox[i][0] + bbox[i][2])
                #cv2.line(frameTrackers,(370,300),(520,300),(255,0,0),5) iphone
                if int(bbox[i][0]) >= 120 and int(bbox[i][0]) <= 250 and int(bbox[i][1]) >= 330 and int(bbox[i][1]) <= 338: #iphone
                #if int(bbox[i][0]) >= 120 and int(bbox[i][0]) <= 250 and int(bbox[i][1]) >= 440 and int(bbox[i][1]) <= 450:  # bayang phone
                        cv2.putText(frameTrackers,(str)(i),(int(bbox[i][0] + 5) ,int(bbox[i][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 0),2)
                        cv2.rectangle(frameTrackers, p1, p2, (0,255,0), 1)
                        if not car_index_count: # empty ang array
                            cv2.line(frameTrackers,(120,330),(250,330),(0,255,0),5)
                            #cv2.line(frameTrackers,(120,420),(250,420),(255,0,0),5)
                            car_index_count[0] = i
                            print "car index : %s"%(i)
                            roi = frame[int(bbox[i][1]):int(bbox[i][1])+int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][0])+int(bbox[i][2])]
                            prev_car_index = current_car_index
                            cv2.imshow('roi',roi)
                            cv2.imwrite('roi-images' + str(counter) + '.png', roi)
                            #print "first coordinates : "
                            #print p1
                            #print "Second coordinates : "
                            #print p2
                        else:
                            cv2.line(frameTrackers,(120,330),(250,330),(0,255,0),5)
                            #cv2.line(frameTrackers,(120,420),(250,420),(255,0,0),5)
                            if i in car_index_count: # already exist sa array
                                #np_delete = np.delete(bbox,i)
                                #print np_delete
                                print "car index : %s"%(i)
                                #print (bbox,i)
                            else: # if wala register
                                counter += 1
                                #print "first coordinates : "
                                #print p1
                                #print "Second coordinates : "
                                #print p2
                                roi = frame[int(bbox[i][1]):int(bbox[i][1])+int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][0])+int(bbox[i][2])]
                                for limit in range(3,-1,-1):
                                    if limit <= 0:
                                        car_index_count[limit] = i
                                    else:
                                        car_index_count[limit] = car_index_count[limit-1]
                                
                                x_coordinate = math.pow(int(bbox[i][0] + bbox[i][2])-int(bbox[i][0]),2)
                                y_coordinate = math.pow(int(bbox[i][1] + bbox[i][3])-int(bbox[i][1]),2)
                                length_cars = math.sqrt(x_coordinate + y_coordinate)
                                cars_count += 1
                                car_saved += 1
                                cv2.imshow('roi',roi)
                                dp_count = dp_count + cars_count
                                #print ("current cars = %d")%cars_count
                                print ("total cars = %d")%dp_count
                                print ("length = %d image number = %d")%(length_cars,counter)
                                if(length_cars > 75):
                                    print p1
                                    print p2
                                    print "car index : %s"%(i)                                
                                    long_vehicle += 1
                                    clong_vehicle += 1
                                    print ("class 2 vehicle = %d")%long_vehicle
                                    total_long = total_long + clong_vehicle
                                    cv2.imwrite('long-images-early-evening' + str(counter) + '.png', roi)
                                    cv2.imshow('roi',roi)
                                else:
                                    short_vehicle += 1
                                    cshort_vehicle += 1
                                    total_short = total_short + cshort_vehicle
                                    print ("class 1 vehicle = %d")%short_vehicle
                                    cv2.imwrite('short-images-early-evening' + str(counter) + '.png', roi)
                                    cv2.imshow('roi',roi)
                                cars_count = 0
                                print car_index_count
                                

                            
                            
            # elif((status[i] == "NEW")):
            #     cv2.putText(frameTrackers,(str)(i),(int(bbox[i][0] + 5) ,int(bbox[i][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 0),2)
            #     cv2.putText(frameTrackers,(str)(status[i][0]),(int(bbox[i][0] + 5) ,int(bbox[i][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 0),2)
            #     print ( "FTrack Text: car: ", i,"tr: ", j, " Col: Blue-Green")
            #     cv2.rectangle(frameTrackers, p1, p2, (255,0,0), 1)
            #     print ("NEW")
            #     pause = True

    time6 = t.time()
    T6 = time6 - time5
    #print("Time6 = ", T6)
    
    totT1 += T1
    totT15 += T15
    totT2 += T2
    totT3 += T3
    totT4 += T4
    totT5 += T5
    totT6 += T6
    """
    print("TotTimes")
    print("1 = ", totT1)
    print("15 = ", totT15)
    print("2 = ", totT2)
    print("3 = ", totT3)
    print("4 = ", totT4)
    print("5 = ", totT5)
    print("6 = ", totT6)"""
    # Display results
    #cv2.putText(frameTrackers,(str)(carCountOut),(int(30 ) ,int(110)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255),3)
    cv2.putText(frameTrackers,(str)(dp_count),(int(50) ,int(60)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255),3)
    cv2.putText(ui_data,(str)(dp_count),(200,100),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1)
    #cv2.putText(frameTrackers,(str)(carCountIn ),(int(530) ,int(110)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255),3)
    cv2.putText(ui_data,(str)(total_long),(200,150),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1) # long
    cv2.putText(ui_data,(str)(total_short),(200,200),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1) # short
    
    cv2.putText(ui_data,(str)(length_cars),(190,350),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1)
    cv2.putText(ui_data,str(total_cars),(200,450),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1)
    cv2.putText(ui_data,str(long_vehicle),(200,500),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1)
    cv2.putText(ui_data,str(short_vehicle),(200,550),cv2.FONT_HERSHEY_SIMPLEX, 0.90,(255, 255, 255),1)
    now_time = datetime.now()
    cv2.putText(ui_data,str(now_time),(200,600),cv2.FONT_HERSHEY_SIMPLEX, 0.30,(255, 255, 255),1)
    #DATABASE SAVE
    elapse = t.time() - start
    cou = int(elapse)
    if(cou % 5 == 0 and cou!= 0):
        save_id = save_id + 1
        save_id = int(save_id)
        #print save_id
        if(save_id == 12):
            current_time = t.strftime('%I:%M:%S:%p')
            curr_time = datetime.now()
            last_time = curr_time + timedelta(seconds=5.0)
            total_cars = long_vehicle + short_vehicle
            if long_vehicle == 0  and short_vehicle == 0:
                saveDatabase(0,long_vehicle,short_vehicle,current_time)
            else:
                total_scars += total_cars
                saveDatabase(total_cars,long_vehicle,short_vehicle,current_time)
            print "total cars = %d"%total_scars
            print "cars counted = %d and id no %d long vehicle %d short vehicle = %s" %(car_saved , save_id, long_vehicle,short_vehicle)
            save_id = 0
            long_vehicle = 0
            short_vehicle = 0
            car_saved = 0
            
    # print ("st = ", status)
    #print "Car in cdo = ", traffic_cdo

    ####### Uncomment These lines to see output of Trackers and Haar
    #In frame Haar red box shows a new detection.
    #The Green Haar Cascades change sizes But the function checkOverlaps checks to see if the detected car is a new detection

    cv2.imshow('Window Result',ui_data)
    #cv2.imshow('Gray',gray)
    cv2.imshow('Result', frameTrackers)
    #cv2.imshow('Haar', frameHaar)
    # cv2.imshow('HaarPrev', frameHaarPrev)
    # cv2.imshow('TrackersPrev', frameTrackersPrev)
    #######

    frameTrackersPrev = frameTrackers.copy()
    # frameHaarPrev     = frameHaar.copy()
    # print ("TrackLife:", trackerLifeTime)
    # print("trackersOn: ", trackersOn)
    # print("carCount: ", carCount)
    # Exit if ESC pressed
    if pause:
        key = cv2.waitKey(1) & 0xFF
        pause = False
        if ((key == ord('q')) | (key == 27)):
            break
    key = cv2.waitKey(1) & 0xFF
    if ((key == ord('q')) | (key == 27)):
        break
    endTime = t.time()
    totalFrames += 1
    # print ("FPS = ", int(1/(endTime-stTime)), "      FPS = ", int(totalFrames/(endTime-absoluteStTime)))
    if(totalFrames == 1000):
        totalFrames = 0
        absoluteStTime = t.time()
    # if(t.time()-absoluteStTime > 20):
    #     break
    stTime = endTime
video.release()
cv2.destroyAllWindows()
