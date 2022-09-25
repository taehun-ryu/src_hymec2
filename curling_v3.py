# 카메라(웹캠) 프레임 읽기 (video_cam.py)
import serial
import cv2
import numpy as np
import time
import math

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수 

wvXXX = np.ones((1,300))
wvXXXvalue = wvXXX * 150

wvR1 = np.arange(1,151) 
wvR = wvR1 / 3
wvL1 = np.flip(wvR)
wvL = wvL1 / 3 

wvX = np.arange(0,300) 
wvY = wvX.reshape(300,1)


wvxxxvaluevalue = (wvX - wvXXXvalue) / 3


moveCase = 0
cntTurn = 0

cntGo = 0
cntRight = 0
cntLeft = 0

gijun = 5000
cntgijun = 50

#목표지점이 카메라에 잡히는지 여부, 잡히면 1 안잡히면 0
goal_on = 0 

#스톤이 카메라에 잡히는지 여부, 잡히면 1 안잡히면 0
stone_on = 1 

#### 스톤, 전수판 중심좌표####
stone_centerX = None
stone_centerY = None

goal_centerX = None 
goal_centerY = None
#############################

goalX = 150  # 예상 목표 좌표 -- 어림잡아 계산

stone = None  # 슛 후 스톤 위치 저장

timeE = time.time()
timeS = 0

shCase = True

goOrRmove = True
goOrLmove = True

port = "COM2"
baud = 115200

ser = serial.Serial(port,baud)


def goleft(): 
    print("goleft")
    ser.write(b'JK0001;')
    time.sleep(0.5)
    global goalX
    goalX +=35

def goright(): 
    print("goright")
    ser.write(b'JK0004;')
    time.sleep(0.5)
    global goalX
    goalX -=34

def gogogo(): 
    print("gogogogogogogogogogogo")
    ser.write(b'JK0002;')
    time.sleep(0.5)

def turnleft(): 
    print("TURNLEFT")
    ser.write(b'JK0080;')
    time.sleep(0.5)
    global goalX
    goalX +=17

def turnright(): 
    print("TURNRIGHT")
    ser.write(b'JK0008;')
    time.sleep(0.5)
    global goalX
    goalX -=15

def SHOOTSHOOT(): 
    print("SHOOT")
    ser.write(b'JK4000;')
    time.sleep(0.5)
    print("TURNRIGHT")
    print("TURNRIGHT")

def slowgo():
    print("slowgo")
    ser.write(b'JK8000;')
    time.sleep(0.5)

    
def nothing(x):
    pass
# 목표지점 HSV찾는 토글바 열기
def settingGoal_bar():
    cv2.namedWindow('HSV_settings')
    cv2.resizeWindow('HSV_settings',400,250)

    cv2.createTrackbar('H_MAX', 'HSV_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MAX', 'HSV_settings', 180)
    cv2.createTrackbar('H_MIN', 'HSV_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MIN', 'HSV_settings', 0)

    cv2.createTrackbar('S_MAX', 'HSV_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('S_MIN', 'HSV_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MIN', 'HSV_settings', 0)

    cv2.createTrackbar('V_MAX', 'HSV_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('V_MIN', 'HSV_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MIN', 'HSV_settings', 0)

# 스톤 HSV찾는 토글바 열기
def settingStone_bar():
    cv2.namedWindow('Stone_settings')
    cv2.resizeWindow('Stone_settings',400,250)

    cv2.createTrackbar('H_MAX', 'Stone_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MAX', 'Stone_settings', 180)
    cv2.createTrackbar('H_MIN', 'Stone_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MIN', 'Stone_settings', 0)

    cv2.createTrackbar('S_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MAX', 'Stone_settings', 225)
    cv2.createTrackbar('S_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MIN', 'Stone_settings', 0)

    cv2.createTrackbar('V_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MAX', 'Stone_settings', 255)
    cv2.createTrackbar('V_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MIN', 'Stone_settings', 240)
   


settingGoal_bar()
settingStone_bar()




cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인

    while True:

        H_maxGoal = cv2.getTrackbarPos('H_MAX', 'HSV_settings')
        H_minGoal = cv2.getTrackbarPos('H_MIN', 'HSV_settings')
        S_maxGoal = cv2.getTrackbarPos('S_MAX', 'HSV_settings')
        S_minGoal = cv2.getTrackbarPos('S_MIN', 'HSV_settings')
        V_maxGoal = cv2.getTrackbarPos('V_MAX', 'HSV_settings')
        V_minGoal = cv2.getTrackbarPos('V_MIN', 'HSV_settings')
  
        H_maxStone = cv2.getTrackbarPos('H_MAX', 'Stone_settings')
        H_minStone = cv2.getTrackbarPos('H_MIN', 'Stone_settings')
        S_maxStone = cv2.getTrackbarPos('S_MAX', 'Stone_settings')
        S_minStone = cv2.getTrackbarPos('S_MIN', 'Stone_settings')
        V_maxStone = cv2.getTrackbarPos('V_MAX', 'Stone_settings')
        V_minStone = cv2.getTrackbarPos('V_MIN', 'Stone_settings')
        
        
        ret, orginimg = cap.read()           # 다음 프레임 읽기
        img = cv2.resize(orginimg, (300, 300))
        
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        #img2 = img.astype(np.uint16)                # dtype 변경 ---①
        #b,g,r = cv2.split(img2)                     # 채널 별로 분리 ---②
        #b,g,r = img2[:,:,0], img2[:,:,1], img2[:,:,2]
        #gray1 = ((b + g + r)/3).astype(np.uint8)    # 평균 값 연산후 dtype 변경 ---③
        
        #gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR을 그레이 스케일로 변경 ---④
        
        # --- ① NumPy API로 바이너리 이미지 만들기
        #thresh_np = np.zeros_like(gray2)   # 원본과 동일한 크기의 0으로 채워진 이미지
        #thresh_np[ gray2 < 127] = 255      # 127 보다 큰 값만 255로 변경
        # ---② OpenCV API로 바이너리 이미지 만들기
        #ret, thresh_cv = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY) 
        
        
        #토츠 쓰레시홀드처리 적응형 임계값 적용 => 오픈씨브이에서 기본으로 제공하는 바이너리, 이진화보다 더 깔끔한 검출, 일일이 임계값을 찾지 않아도 됨.
        #t, t_otsu = cv2.threshold(gray2, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        
        #가우시안블러쓰레시홀드처리 
        #th3 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                             cv2.THRESH_BINARY, blk_size, C)
        
############################################################################

        # 하이퍼 파라미터를 줄이는게 좋다
        kernal_op = np.ones((8, 8), "uint8")    # motph_open을 위한 커널, 커널의 크기를 지정해 줘야함
        kernal_cl = np.ones((11, 11), "uint8")  # morph_close를 위한 커널
        
        # 스톤 HSV
        stone_lower = np.array([H_minStone, S_minStone, V_minStone], np.uint8)
        stone_upper = np.array([H_maxStone, S_maxStone, V_maxStone], np.uint8) 
        stone_mask = cv2.inRange(hsvFrame, stone_lower, stone_upper)
        
        res_stone = cv2.bitwise_and(img, img, mask = stone_mask)
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, kernal_cl,iterations=3) # 내부 노이즈(검은 구멍) 제거(메움)
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_OPEN, kernal_op,iterations=2) # 외부 노이즈(점으로 보이는 노이즈)를 제거

        # 목표지점(빨,노,초) HSV 
        goal_lower = np.array([H_minGoal, S_minGoal, V_minGoal], np.uint8) 
        goal_upper = np.array([H_maxGoal, S_maxGoal, V_maxGoal], np.uint8) 
        goal_mask = cv2.inRange(hsvFrame, goal_lower, goal_upper) 

        # for goal 
        res_goal = cv2.bitwise_and(img, img, mask = goal_mask)
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_OPEN, kernal_op,iterations=2) 
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_CLOSE, kernal_cl,iterations=5)

        ############################################################################
        
        ########### 이진화 한 값을 0과 255를 바꿈 (그... 지금 하고 있는곳이 하얀색 배경에 검정색 그림이기 때문)
        thresh = stone_mask
        ########너무 큰 숫자를 만들지 않기 위해 0 255를 0 1 로 바꿈 (행렬구조임)
        thresh_02 = thresh/255
        thresh_01 = thresh_02 * wvY 
        ######### 이 행렬구조의 원소들의 합을 구함
        ###########열을 다해서 행으로 만듦
        threshrowsum1 = np.sum(thresh_01, axis = 0)
        
        threshrowsum = threshrowsum1 * wvxxxvaluevalue * wvxxxvaluevalue* wvxxxvaluevalue
        
        ##############그 열안에 있는 원소들의 합을 다 함.
        threshallsum = (np.sum(threshrowsum, axis=1))/1000000



############################################################################
        contours, hierarchy = cv2.findContours(stone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
        # 꼭지점 좌표만을 갖는 컨투어 그리기, stone
        cv2.drawContours(img, contours, -1, (0,0,255), 4)

        if(contours == ()):     # 스톤이 보이는지 판단
            stone_on = 0        # 스톤 안 보임
        else:
            stone_on = 1        # 스톤 보임

        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour)
            _,radius = cv2.minEnclosingCircle(contour)

            ratio = radius * radius * math.pi / area

            if int(ratio) == 1:
                #print("Cir")
                pass

            if(area > 200): #stone_mask 영역에 따른 조건.
               x, y, w, h = cv2.boundingRect(contour)
               stone_centerX = int((x+x+w)/2)
               stone_centerY = int((y+y+h)/2)
               img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) #사각형 그리기.
               img = cv2.circle(img, (stone_centerX, stone_centerY), 2, (0, 255, 0), 2)
              
               cv2.putText(img, "Stone", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255))

      

############################################################################
        img = cv2.circle(img, (goalX,10), 2, (0, 255, 0), 2)
        img = cv2.line(img, (goalX,10), (150,300), (0,0,0), 2)
        print(stone_centerX)
############################################################################

        # goal_mask의 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환
        contours, hierarchy = cv2.findContours(goal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 꼭지점 좌표만을 갖는 컨투어 그리기, goal
        cv2.drawContours(img, contours, -1, (0,255,0), 4)

        if(contours == ()):     # 목표지점이 보이는지 판단
            goal_on = 0        # 목표지점 안 보임
            #print(goal_on," , ",stone_centerY)
        else:
            goal_on = 1        # 목표지점 보임
            #print(goal_on," , ",stone_centerY) 
  
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 100):  #goal_mask 영역에 따른 조건.

              x, y, w, h = cv2.boundingRect(contour) 
              img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
              goal_centerX = int((x+x+w)/2)
              goal_centerY = int((y+y+h)/2)  
              img = cv2.circle(img, (goal_centerX, goal_centerY), 2, (0, 255, 0), 2)
  
              cv2.putText(img, "Goal place", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

############################################################################
        
        '''
        ##### 반갈라서 왼쪽의 합을 구함
        leftthress = thresh_01[0:300, 0:150]
        leftthressrowsum = np.sum(leftthress, axis=0)
        leftthressrowsumWv = leftthressrowsum * wvL * wvL
        leftthresssumsum = np.sum(leftthressrowsumWv, axis= 0)
        
        ###############반갈라서 오른쪽 합을 구함
        rightthress = thresh_01[0:300, 150:300]
        rightthressrowsum = np.sum(rightthress, axis=0)
        rightthressrowsumWv = rightthressrowsum * wvR * wvR
        rightthresssumsum = np.sum(rightthressrowsumWv,axis=0)
        
        ####################150칸짜리 가중치를 위한 열을 만듦
        '''
        stoneDetect =  int(threshallsum)
        
        #print(int(threshallsum) )
        
        #print(rightthresssumsum + leftthresssumsum)
        
        #헤리스 코너 응답함ㅅ 계산
        # 좋은 특징점 검출 방법
        #corners = cv2.goodFeaturesToTrack(t_otsu, 400, 0.1, 10)

        #dst1 = cv2.cvtColor(t_otsu, cv2.COLOR_GRAY2BGR)
        '''
        if corners is not None:
            for i in range(corners.shape[0]): # 코너 갯수만큼 반복문
                pt = (int(corners[i, 0, 0]), int(corners[i, 0, 1])) # x, y 좌표 받아오기
                cv2.circle(dst1, pt, 5, (0, 0, 255), 2) # 받아온 위치에 원
       
        
        #fast 코너 검출
        fast = cv2.FastFeatureDetector_create(60) # 임계값 60 지정
        keypoints = fast.detect(gray2) # Keypoint 객체를 리스트로 받음

        dst2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1])) # kp안에 pt좌표가 있음
            cv2.circle(dst2, pt, 5, (0, 0, 255), 2)      
        '''
        timeS = time.time()

        

        if (moveCase == 0):
            if(shCase):
                SHOOTSHOOT()
                shCase = False

            stone = stone_centerX
            gogogo()

            if(timeS - timeE > 2):
                moveCase = 1
                shCase = True
                timeE = time.time()
            

        elif (moveCase == 1):   # 스톤 앞까지 전진
            if(stone_on):

                if (-gijun < stoneDetect < gijun):
                    cntGo = cntGo+1

                elif(stoneDetect > gijun):
                    cntRight = cntRight+1
                elif(stoneDetect < -gijun):
                    cntLeft = cntLeft + 1

                if (cntGo > cntgijun ):
                    slowgo()
                    time.sleep(0.01)
                    cntGo = 0
                    cntLeft = 0
                    cntRight = 0

                elif(cntLeft > cntgijun):
                    turnleft()
                    time.sleep(0.01)
                    cntTurn -= 1
                    cntGo = 0
                    cntLeft = 0
                    cntRight = 0

                elif (cntRight > cntgijun):
                    turnright()
                    time.sleep(0.01)
                    cntTurn += 1
                    cntGo = 0
                    cntLeft = 0
                    cntRight = 0

                if(stone_centerY >= 255):
                    moveCase = 2
                    timeE = time.time()

            else:  # 스톤 안보이면
                if(stone_centerX>160):
                    if(timeS - timeE >1 and goOrRmove):
                        if(goOrRmove):
                            goright()
                            goright()
                            goright()
                            goright()
                            goOrRmove = False
                            
                        else:
                            while(1):
                                gogogo()
                                if(stone_on):
                                    break
                            goOrRmove = True
                            

                elif(stone_centerX<140):
                    if(timeS - timeE >1 and goOrLmove):
                        if(goOrLmove):
                            goleft()
                            goleft()
                            goOrLmove = False

                        else:
                            while(1):
                                gogogo()
                                if(stone_on):
                                    break
                            goOrLmove = True

                elif(stone_centerX>=140 and stone_centerX<=160):
                    while(1):
                        gogogo()
                        if(stone_on):
                            break
                    goOrLmove = True

                timeE = time.time()
        
        elif (moveCase == 2):  # 각도 조절

            if(stone>=160):
                if (cntTurn == 0 and timeS - timeE>1):
                    moveCase = 3

                elif(cntTurn > 0 and timeS - timeE>1):
                    for i in range(math.floor(cntTurn)):
                        turnleft()
                
                    moveCase = 3

                elif(cntTurn < 0 and timeS - timeE>2):
                    for i in range(math.ceil(cntTurn)):
                        turnright()
                        
                    moveCase = 3
                timeE = time.time()

            if(stone<=140):
                if (cntTurn == 0 and timeS - timeE>1):
                    moveCase = 3

                elif(cntTurn > 0 and timeS - timeE>1):
                    for i in range(math.floor(cntTurn/2)):
                        turnleft()
                
                    moveCase = 3

                elif(cntTurn < 0 and timeS - timeE>1):
                    for i in range(math.ceil(-cntTurn/2)):
                        turnright()
                    moveCase = 3
                timeE = time.time()

        elif(moveCase ==3):  # 목표지점 바라보기

            if(goalX>=145 and goalX <=155):
                moveCase ==4
                timeE = time.time()
                
            elif(goalX<145):
                turnleft()

            elif(goalX>=155):
                turnright()
            
            

        # x = 220 , y = 260 일 때 오른발 앞에 있음
        
        elif (moveCase == 4):  # 오른발 - 스톤 위치 조정

            if(timeS - timeE >1):
                if(stone_centerY<260):
                    moveCase =1

                else:
                    if(stone_centerX>225):
                        goright()

                    elif(stone_centerX<215):
                        goleft()

                    elif(stone_centerX>=215 and stone_centerX <=225):
                        moveCase = 0


        

            

            
            
            
            
        if ret:
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            #cv2.imshow('gray1', gray1)
            #cv2.imshow('gray2', gray2)
            #cv2.imshow('thresh_cv', thresh_cv)
            #cv2.imshow('t_otsu', thresh)
            cv2.imshow('th3', stone_mask)
            #cv2.imshow('dst1', dst1)
            #cv2.imshow('dst2', dst2)
            
            
            if cv2.waitKey(1) != -1:    # 1ms 동안 키 입력 대기 ---②
                break                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break


else:
    print("can't open camera.")
cap.release()                           # 자원 반납
cv2.destroyAllWindows()