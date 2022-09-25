from turtle import left
import numpy as np 
import cv2
import math
import time
import serial


webcam = cv2.VideoCapture(0) 

#전역변수 설정

wvX = np.arange(0,300) 
wvY = wvX.reshape(300,1)


stone_centerX = None   #스톤 좌표 전역설정
stone_centerY = None

goal_centerX = None   #목표 좌표 전역설정
goal_centerY = None

#스톤이 카메라에 잡히는지 여부, 잡히면 1 안잡히면 0
stone_on = 1  

#목표지점이 카메라에 잡히는지 여부, 잡히면 1 안잡히면 0
goal_on = 0 

lrCase = 0  #게걸음

moveCase =0 #움직임케이스

lineCheck = None

timeS = 0
timeE = 0

port = "COM1"
baud = 115200

#ser = serial.Serial(port,baud)

def goleft(): 
    print("goleft")
    #ser.write(b'JK0001;')
    #time.sleep(1)

def goright(): 
    print("goright")
    #ser.write(b'JK0004;')
    #time.sleep(1)

def gogogo(): 
    print("gogogogogogogogogogogo")
    #ser.write(b'JK0002;')
    #time.sleep(1)

def turnleft(): 
    print("TURNLEFT")
    #ser.write(b'JK0080;')
    #time.sleep(1)

def turnright(): 
    print("TURNRIGHT")
    #ser.write(b'JK0008;')
    #time.sleep(1)

def SHOOTSHOOT(): 
    print("SHOOT")
    #ser.write(b'JK1000;')
    #time.sleep(1)


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
    cv2.setTrackbarPos('H_MAX', 'Stone_settings', 115)
    cv2.createTrackbar('H_MIN', 'Stone_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MIN', 'Stone_settings', 93)

    cv2.createTrackbar('S_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MAX', 'Stone_settings', 222)
    cv2.createTrackbar('S_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MIN', 'Stone_settings', 122)

    cv2.createTrackbar('V_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MAX', 'Stone_settings', 255)
    cv2.createTrackbar('V_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MIN', 'Stone_settings', 219)
   


settingGoal_bar()
settingStone_bar()


while(1):
    try:

        _, imageFrame = webcam.read()

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
  
  
        
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
  
        # 스톤 HSV
        stone_lower = np.array([H_minStone, S_minStone, V_minStone], np.uint8)
        stone_upper = np.array([H_maxStone, S_maxStone, V_maxStone], np.uint8) 
        stone_mask = cv2.inRange(hsvFrame, stone_lower, stone_upper)
  
  
        # 목표지점(빨,노,초) HSV 
        goal_lower = np.array([H_minGoal, S_minGoal, V_minGoal], np.uint8) 
        goal_upper = np.array([H_maxGoal, S_maxGoal, V_maxGoal], np.uint8) 
        goal_mask = cv2.inRange(hsvFrame, goal_lower, goal_upper) 
  
  
        # 하이퍼 파라미터를 줄이는게 좋다
        kernal_op = np.ones((8, 8), "uint8")    # motph_open을 위한 커널, 커널의 크기를 지정해 줘야함
        kernal_cl = np.ones((11, 11), "uint8")  # morph_close를 위한 커널
  
        # For stone  
        res_stone = cv2.bitwise_and(imageFrame, imageFrame, mask = stone_mask)
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, kernal_cl,iterations=3) # 내부 노이즈(검은 구멍) 제거(메움)
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_OPEN, kernal_op,iterations=5) # 외부 노이즈(점으로 보이는 노이즈)를 제거
  
        # for goal 
        res_goal = cv2.bitwise_and(imageFrame, imageFrame, mask = goal_mask)
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_OPEN, kernal_op,iterations=2) 
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_CLOSE, kernal_cl,iterations=5) 
  
        
  
  
        # stone_mask의 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환
        contours, hierarchy = cv2.findContours(stone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
        # 꼭지점 좌표만을 갖는 컨투어 그리기, stone
        cv2.drawContours(imageFrame, contours, -1, (0,0,255), 4)

        if(contours == ()):     # 스톤이 보이는지 판단
            stone_on = 0        # 스톤 안 보임
            #print(stone_on," , ",stone_centerY)
        else:
            stone_on = 1        # 스톤 보임
            #print(stone_on," , ",stone_centerY)

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
               imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2) #사각형 그리기.
               imageFrame = cv2.circle(imageFrame, (stone_centerX, stone_centerY), 2, (0, 255, 0), 2)
              
               cv2.putText(imageFrame, "Stone", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255))
        
  
        # goal_mask의 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환
        contours, hierarchy = cv2.findContours(goal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 꼭지점 좌표만을 갖는 컨투어 그리기, goal
        cv2.drawContours(imageFrame, contours, -1, (0,255,0), 4)

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
              imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              goal_centerX = int((x+x+w)/2)
              goal_centerY = int((y+y+h)/2)  
              imageFrame = cv2.circle(imageFrame, (goal_centerX, goal_centerY), 2, (0, 255, 0), 2)
  
              cv2.putText(imageFrame, "Goal place", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))


        cam_line = cv2.line(imageFrame,(320,480),(320,0),(0,0,0), 2)
  
  
        #resize
        goal_mask_resize = cv2.resize(goal_mask,(300,300))
        res_goal_resize = cv2.resize(res_goal,(300,300))
        stone_mask_resize = cv2.resize(stone_mask,(300,300))
        res_stone_resize = cv2.resize(res_stone,(300,300))
       
  
        # openCV pixel = 640 x 480
        # Program Termination
        cv2.imshow("cam", imageFrame)
        cv2.imshow("goal binary", goal_mask_resize)
        cv2.imshow("goal res", res_goal_resize)
        cv2.imshow("stone binary", stone_mask_resize)
        cv2.imshow("stone res", res_stone_resize)
        
  
  
        if cv2.waitKey(10) & 0xFF == ord('q'): 
           #cap.release() #이거 왜 필요하지?
           cv2.destroyAllWindows() 
           break

        bigyo = np.equal(goal_mask,stone_mask)
        result = np.where(bigyo==True,1,0)

        sum1 = np.sum(result,axis=0)
        print(sum1)
        
  
        ######### 제어코드 #########
  
        # 고개를 상하좌우로 움직일 수 있다
        timeS = time.time()

        if(moveCase==0):
            SHOOTSHOOT()
            timeE = time.time()
            moveCase=1
        
        elif(moveCase==1):

            if(not stone_on):  #스톤이 카메라에 안 잡히면
                print("neck up")  # 고개 들기
                moveCase = 2
                timeE=time.time()

            else:           #스톤이 카메라에 잡히면
                if(stone_centerY<450):   # 스톤이 멀리 있으면
                    if(timeS-timeE>2):
                        gogogo()
                        timeE=time.time()
                else:
                    if(stone_centerX<310 and timeS - timeE >2):  #스톤이 왼쪽에 있으면
                        lrCase = 1
                        moveCase =3
                        timeE = time.time()
  
  
                    elif(stone_centerX>330 and timeS - timeE >2):  #스톤이 오른쪽에 있으면
                        lrCase = 2
                        moveCase = 3
                        timeE = time.time()

        elif(moveCase==2): #고개 든 상태로

            if(stone_on and stone_centerY>=240):  #스톤이 있고 스톤이 화면상 절반보다 아래에 위치하면

                if(timeS - timeE >2):

                    print("neck down")
                    moveCase=1
                    timeE = time.time()

            elif((not stone_on and timeS - timeE >2) or (stone_centerY<240 and timeS-timeE>2)):
                # 스톤이 화면에 안잡히거나 화면상 절반보다 더 위에 있어면
                gogogo()
                timeE = time.time()

        elif(moveCase==3):

            if(stone_centerY>=450 and lrCase==1 and timeS - timeE >2):  # 왼쪽에 공이 있으면

                goleft()
                timeE = time.time()

                if(stone_centerX>=325):  #왼쪽으로 가다가 공이 앞에 있으면

                    if(goal_on):  # 목표지점 보이면
                        moveCase=4
                    else:         # 목표지점 안 보이면
                        moveCase=5
                        timeE = time.time()

            elif(stone_centerY>=450 and lrCase==2 and timeS-timeE>2): #오른쪽에 공이 있으면

                goright()
                timeE = time.time()

                if(stone_centerX>=325):  #오른쪽으로 가다가 공이 앞에 있으면

                    if(goal_on):
                        moveCase=4
                    else:
                        moveCase=5
                        timeE = time.time()

            elif(stone_centerY<=450):  # 공에서 부득이하게 멀어지면

                moveCase=1
                    

        elif(moveCase==4):  # 스톤과 자신과 목표지점이 일자가 되도록 회전
  
            #라인 일치 판단
            if( ((320-stone_centerX) != 0) and ((goal_centerX-stone_centerX) != 0)):
               # 로봇 - 스톤 - 목표지점 이 이루는 각도 계산
               f_d_1 = math.degrees(np.arctan((stone_centerY-0)/(stone_centerX-320)))
               f_d_2 = math.degrees(np.arctan((goal_centerY-stone_centerY)/(goal_centerX-stone_centerX)))
   
               #로봇 - 스톤 - 목표지점이 이루는 각도
               robot_stone_goal = f_d_1 - f_d_2
   
               #print(robot_stone_goal)
   
               if(robot_stone_goal>=-5 and robot_stone_goal<=5):
                  lineCheck = 1
               elif(robot_stone_goal>=5):   #왼쪽으로 돌아야 함.
                  lineCheck = 2     
               elif(robot_stone_goal<=-5):  #오른쪽으로 돌아야 함.
                  lineCheck = 3
               
            else:
               lineCheck = None
            
            
            if(lineCheck == 2 and timeS - timeE >2 ):
               goleft()
               time.sleep(0.5)
               turnright()
               timeE = time.time()
   
   
            elif(lineCheck == 3 and timeS - timeE >2 ):
               goright()
               time.sleep(0.5)
               turnleft()
               timeE = time.time()
   
            elif(lineCheck == 1):
               moveCase = 5
               timeE = time.time()
 
        elif(moveCase ==5 and timeS - timeE >2):
            SHOOTSHOOT()
            moveCase = 1
            time.sleep(7)
            

            
            


        

    except TypeError:
      pass