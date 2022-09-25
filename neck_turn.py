import numpy as np
import serial
import time

port = "COM2"
baud = 115200
ser = serial.Serial(port,baud)

global angle_RL

global neckrlcnt

global angle_UD

global neckUDcnt

global half

angle_UD = 30000

angle_RL = 30000

neckrlcnt = 0 

half = False


def neckTrunLR(sign,angle_RL_value):   # 왼쪽 or 오른쪽
    global angle_RL

    if sign==0:  # left
        print("neckrleft")
        angle_RL = angle_RL + angle_RL_value

    elif sign==1:  # right
        print("neckright")
        angle_RL = angle_RL - angle_RL_value

    ser.write(b'PS00023,')
    ser.write(bytes(str(angle_RL),encoding='ascii'))
    ser.write(b';')
    time.sleep (0.1)

def neckTurnUD(sign,angle_UD_value):  # 위 or 아래
    global angle_UD

    if sign==0:  # 아래
        print("neckdown")
        angle_UD = angle_UD + angle_UD_value

    elif sign==1:  # 위
        print("neckup")
        angle_UD = angle_UD - angle_UD_value

    ser.write(b'PS00024,')
    ser.write(bytes(str(angle_UD),encoding='ascii'))
    ser.write(b';')
    time.sleep (0.1)


def neckTurnMain():  # 왼 - 위 - 오 - 위
    global angle_RL
    global angle_UD

    global half

    if  half:
        neckTrunLR(1,1000)
        if angle_RL < 25000 :
            neckTurnUD(1,1000)
            half = False
            time.sleep(0.1)

    if not half:
        neckTrunLR(0,1000)
        if angle_RL > 35000 :
            neckTurnUD(1,1000)
            half = True
            time.sleep(0.1)
while True:
    neckTurnMain()


'''
공이 화면 영역안에서 사라지면 100이나 200씩 내려사 800정도까지 내리면 될 듯.
'''