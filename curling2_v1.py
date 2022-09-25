import serial 
import cv2
import numpy as np
import time
import math

# port = "COM2"
# baud = 115200
# ser = serial.Serial(port,baud)

######################
stone_centerX = None
stone_centerY = None
goal_centerX = None
goal_centerY = None
sx = None
sy = None
sw = None
sh = None
gx = None
gy = None
gw = None
gh = None
######################
stone_location = None
case1_judgement = True
case = 0
overlap_check = 0
######################
wvXXX = np.ones((1,300))
wvXXXvalue = wvXXX * 150

wvR1 = np.arange(1,151) 
wvR = wvR1 / 3
wvL1 = np.flip(wvR)
wvL = wvL1 / 3 

wvX = np.arange(0,300) 
wvY = wvX.reshape(300,1)


wvxxxvaluevalue = (wvX - wvXXXvalue) / 3
######################
global angle_RL

global angle_UD

angle_UD = 30000

angle_RL = 30000


def nothing(x):
    pass

def settingGoal_bar():
    cv2.namedWindow('Goal_settings')
    cv2.resizeWindow('Goal_settings',400,250)

    cv2.createTrackbar('H_MAX', 'Goal_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MAX', 'Goal_settings', 180)
    cv2.createTrackbar('H_MIN', 'Goal_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MIN', 'Goal_settings', 0)

    cv2.createTrackbar('S_MAX', 'Goal_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MAX', 'Goal_settings', 255)
    cv2.createTrackbar('S_MIN', 'Goal_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MIN', 'Goal_settings', 0)

    cv2.createTrackbar('V_MAX', 'Goal_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MAX', 'Goal_settings', 255)
    cv2.createTrackbar('V_MIN', 'Goal_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MIN', 'Goal_settings', 0)

# 스톤 HSV찾는 토글바 열기
def settingStone_bar():
    cv2.namedWindow('Stone_settings')
    cv2.resizeWindow('Stone_settings',400,250)

    cv2.createTrackbar('H_MAX', 'Stone_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MAX', 'Stone_settings', 174)
    cv2.createTrackbar('H_MIN', 'Stone_settings', 0, 180, nothing)
    cv2.setTrackbarPos('H_MIN', 'Stone_settings', 93)

    cv2.createTrackbar('S_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MAX', 'Stone_settings', 245)
    cv2.createTrackbar('S_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('S_MIN', 'Stone_settings', 111)

    cv2.createTrackbar('V_MAX', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MAX', 'Stone_settings', 199)
    cv2.createTrackbar('V_MIN', 'Stone_settings', 0, 255, nothing)
    cv2.setTrackbarPos('V_MIN', 'Stone_settings', 156)
   
def neckTrunLR(angle_RL_value):   # 왼쪽 or 오른쪽
    global angle_RL
    print(angle_RL)
    angle_RL = angle_RL + angle_RL_value
    if (angle_RL < 25000):
        angle_RL = 25000
    elif(angle_RL>35000):
        angle_RL = 35000
    ser.write(b'PS00023,')
    ser.write(bytes(str(angle_RL),encoding='ascii'))
    ser.write(b';')
    time.sleep (0.01)

def neckTrunUD(angle_UD_value):   # 위 or 아래
    global angle_UD
    angle_UD = angle_UD + angle_UD_value
    if(angle_UD < 30000):
        angle_UD = 30000
    elif(angle_UD > 35000):
        angle_UD = 35000
    ser.write(b'PS00024,')
    ser.write(bytes(str(angle_UD),encoding='ascii'))
    ser.write(b';')
    time.sleep (0.01)
    print(angle_UD)


settingGoal_bar()
settingStone_bar()


cap = cv2.VideoCapture(2,cv2.CAP_V4L2)   # Window: 0 or 1 , Ubuntu: 2

if cap.isOpened():                      # 캡쳐 객체 연결 확인

    while True:

        H_maxGoal = cv2.getTrackbarPos('H_MAX', 'Goal_settings')
        H_minGoal = cv2.getTrackbarPos('H_MIN', 'Goal_settings')
        S_maxGoal = cv2.getTrackbarPos('S_MAX', 'Goal_settings')
        S_minGoal = cv2.getTrackbarPos('S_MIN', 'Goal_settings')
        V_maxGoal = cv2.getTrackbarPos('V_MAX', 'Goal_settings')
        V_minGoal = cv2.getTrackbarPos('V_MIN', 'Goal_settings')
  
        H_maxStone = cv2.getTrackbarPos('H_MAX', 'Stone_settings')
        H_minStone = cv2.getTrackbarPos('H_MIN', 'Stone_settings')
        S_maxStone = cv2.getTrackbarPos('S_MAX', 'Stone_settings')
        S_minStone = cv2.getTrackbarPos('S_MIN', 'Stone_settings')
        V_maxStone = cv2.getTrackbarPos('V_MAX', 'Stone_settings')
        V_minStone = cv2.getTrackbarPos('V_MIN', 'Stone_settings')
        
        
        ret, orginimg = cap.read()           # 다음 프레임 읽기
        img = cv2.resize(orginimg, (300, 300))

        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

        contours, hierarchy = cv2.findContours(stone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # goal_mask에 해당하는 HSV값을 가지는 픽셀의 중심점
        G = cv2.moments(goal_mask,True)

        if int(G['m00'])!=0:
            # 실직적인 진짜 목표지점의 중심좌표
            gcx = int(G['m10']/G['m00'])
            gcy = int(G['m01']/G['m00'])

            #print(cx," , ",cy )
            img = cv2.circle(img, (gcx, gcy), 2, (255, 0, 0), 2)
        
        # 꼭지점 좌표만을 갖는 컨투어 그리기, stone
        cv2.drawContours(img, contours, -1, (0,0,255), 4)

        if(contours == ()):     # 스톤이 보이는지 판단

            stone_on = 0        # 스톤 안 보임
        else:

            stone_on = 1        # 스톤 보임

        for pic, contour in enumerate(contours): 

            area = cv2.contourArea(contour)
            _,radius = cv2.minEnclosingCircle(contour)

            if(area > 200): #stone_mask 영역에 따른 조건.
               sx, sy, sw, sh = cv2.boundingRect(contour)
               stone_centerX = int((sx+sx+sw)/2)
               stone_centerY = int((sy+sy+sh)/2)
               img = cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2) #사각형 그리기.
               img = cv2.circle(img, (stone_centerX, stone_centerY), 2, (0, 0, 255), 2)
               cv2.putText(img, "Stone", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255))
        
        ########### 이진화 한 값을 0과 255를 바꿈 (그... 지금 하고 있는곳이 하얀색 배경에 검정색 그림이기 때문)
        thresh = stone_mask
        ########너무 큰 숫자를 만들지 않기 위해 0 255를 0 1 로 바꿈 (행렬구조임)
        thresh_02 = thresh/255
        thresh_01 = thresh_02 * wvY 
        ######### 이 행렬구조의 원소들의 합을 구함
        ###########열을 다해서 행으로 만듦
        threshrowsum1 = np.sum(thresh_01, axis = 0)
        threshcorsum1 = np.sum(thresh_01, axis = 1)
        threshcorsum = threshcorsum1 * wvxxxvaluevalue * wvxxxvaluevalue* wvxxxvaluevalue
        threshrowsum = threshrowsum1 * wvxxxvaluevalue * wvxxxvaluevalue* wvxxxvaluevalue
        
        ##############그 열안에 있는 원소들의 합을 다 함.
        threshallsum = (np.sum(threshrowsum, axis=1))/100000000
        threshallsum1 = (np.sum(threshcorsum, axis=1))/100000000

        
        lineDetectx =  int(threshallsum)
        lineDetecty =  int(threshallsum1)

        # goal_mask의 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환
        contours, hierarchy = cv2.findContours(goal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 꼭지점 좌표만을 갖는 컨투어 그리기, goal
        cv2.drawContours(img, contours, -1, (0,255,0), 4)

        if(contours == ()):     # 목표지점이 보이는지 판단
            goal_on = 0        # 목표지점 안 보임
            
        else:
            goal_on = 1        # 목표지점 보임
  
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 100):  #goal_mask 영역에 따른 조건.

              gx, gy, gw, gh = cv2.boundingRect(contour) 
              img = cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
              goal_centerX = int((gx+gx+gw)/2)
              goal_centerY = int((gy+gy+gh)/2)  
              img = cv2.circle(img, (goal_centerX, goal_centerY), 2, (0, 255, 0), 2)
  
              cv2.putText(img, "Goal place", (gx, gy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

        # Rectamgle Overlap Check
        sum = 0
        if  sx is not None and gx is not None:
            if (sx <= gx+gw) and (sx+sw >= gx):
                sum += 1

            if (sy <= gy+gh) and (sh+sh >= gy):
                sum += 1

            if sum == 2:
                overlap_check = 1
                # print(overlap_check,": Overlap")
            else:
                overlap_check = 0
                # print(overlap_check,":No overlap")

        ######### 제어코드 #########
        '''
        0. 슛 한다.
            * 후에 목을 들기.

        1. 좌우 맞추기
            * 공이 휘었을 경우(너무 많이 휘었으면 애초에 모션 문제) 어떤 조건에 근거하여 좌-우 움직임을 줄 것인가?
                a. 공이 화면 정중앙에 오도록 목을 돌린 후 고개가 돌아간 쪽으로 움직인다. 계속 반복 - 언제까지?
                b. 목 엔코더가 정면이고 공이 중앙에 있을 때 까지 - 짧은 옆걸음도 필요할거 같아. 기검이랑 얘기해보자.
                c. 이젠 공이랑 일자가 됨.  - 하지만 마지막 슛 동작에서 목표지점과 일자로 맞추는 게 관건이 될 듯.
            * 공이 일자로 갔다면?
                a. 베스트지. 그냥 직진하면 될거 같아.

        2. 공 쪽으로 가기
            * 일단 1번 과정을 통해 로봇의 직진경로에 공이 있는 상황에서 직진을 할 텐데, 어떤 조건에 근거하여 멈출 것인가?
                a. 직진하다가 공이 사라지면 고개를 아래로 내린다. 저번에 측정해 봤을 때 100~200정도 내리면 됐던거 같아.
                b. 공이 발 앞에 올 때 까지 반복 - 공이 발 앞에 왔다는 걸 어떻게 알 것인가?
                c. 공이 로봇의 발 앞에 있을 때의 목 각도, 화면의 픽셀좌표를 알아내서 해당 목각도에서 해당 픽셀 좌표에 왔는지 판단.
                d. 이제 공이 로봇 바로 앞까지 왔다.

                * 완벽하게 직진하지 못해 중간에 공이 직진 경로에서 벗어나면?
                    a. 제자리 턴 해서 다시 맞춘다. - 직진경로에서 벗어나는 지를 계속 체크하면서 앞으로 가야할 듯.

                * 공을 발 앞에 놓긴 했는데 목표지점의 중앙 <> 스톤 <> 로봇이 일자가 아니면? 처음 슛에서 날아간 공이 일자 경로에 없으면
                   이럴 수 있다. - 음......
                    a. 하아...........

        ** 거리를 먼저 맞출지, 좌-우를 먼저 맞출지, 거리-좌우-거리-좌우 로 할 지 생각 좀 해야 할 듯.

        *** 일단 여기까지 성공할 수 있는지 보자. 이거 안되면 뒤에건 필요없음. 알고리즘을 다시 짜야하는 거라서...

        3. 디테일 작업
            * 2번 과정을 통해 공이 로봇의 바로 앞에 위치한 상황. 공을 일자로 차기 위해선 정확한 타격지점을 잡아야 할 거 같은데?
                a.  짧은 옆걸음으로 맞출까? - 얼만큼 짧게가 가능하지? - 기검이랑 얘기해 봐야한다.

            * 끝나면 0번으로 회귀
                a. 종료 조건에 맞으면 회귀 X -- 이건 있는게 오히려 독일 수도 있다. 정말 나중에 사용
                b. 일단은 우리가 강제종료 시키자. 티 안나게
        
        **. 목표지점 <> 스톤 <> 로봇 은 일자로 맞출거야? - 음......
            * 중앙으로 차는거에 목적을 두지 않을지도.
            * 슛카운트가 2 or 3이상일 때 공과 목표지점이 overlap 되면 종료 -- 공 차는 비거리가 생각보다 중요할 지도?
            
        '''
        max_time = time.time()
        # print('sx: ',stone_centerX,' , ',stone_centerY)

        if case ==0:   # 일단 슈웃

            print("shoot")
            print("neck up")
            time.sleep(2)
            case =11

        elif case ==11:   # 공찾기
            if stone_centerX is not None and stone_centerY is not None:
                print('공찾기: ',stone_centerX)
                # neckTrunLR(-lineDetectx)
                # neckTrunUD(lineDetecty)

                if 145<=stone_centerX and stone_centerX<=155:
                    case=12
                else:
                    pass
            else:
                pass

        elif case ==12:  # 옆걸음
            if stone_on:

                if angle_RL < 30000:
                    for i in range(2):
                        print('왼쪽 옆걸음')
                        time.sleep(2)

                    case =11

                elif angle_RL > 30000:
                    for i in range(2):
                        print('오른쪽 옆걸음')
                        time.sleep(2)
                        
                    case=11
                
                if angle_RL==30000 and (145<stone_centerX and stone_centerX<155):
                    case =21
            else:
                pass

        elif case ==21:   # 앞으로 가기
            if stone_on:
                for i in range(1):
                    print('gogogogogoggogoggogogo')

                if stone_centerY<50:
                    case = 22
                
                if not(145<stone_centerX and stone_centerX<155):
                    time.sleep(0.1)
                    case = 11
            else:
                pass

        elif case ==22:   # 목 내리기
            if stone_on:
                print('목 내리기: 100')
                # neckTrunUD(-100)
                time.sleep(1)
                
                if angle_UD==25000 and (145<stone_centerX and stone_centerX<155):
                    case =3

                elif angle_UD==25000 and (stone_centerX<145):

                    for i in range(1):
                        print('왼쪽 옆걸음')

                elif angle_UD==25000 and (155<stone_centerX): 
                    for i in range(1):
                        print('오른쪽 옆걸음')
                
                elif not angle_UD==25000:
                    case=21
            else:
                pass

        elif case ==3:
            print('성공성공성공성공성공성공성공성공성공성공성공성공성공')

        if ret:
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            cv2.imshow('goal_mask',goal_mask)
            cv2.imshow('res_goal',res_goal)
            cv2.imshow('stone_mask',stone_mask)
            cv2.imshow('res_stone',res_stone)
                
                
            if cv2.waitKey(1) != -1:    # 1ms 동안 키 입력 대기 ---②``
                break                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break