import numpy as np
import serial
import time

port = "COM2"
baud = 115200
ser = serial.Serial(port,baud)

ser.write(b'PS00023;')  # byte 형태로 변환해서 write(전송)

read_motors = ser.readlines()

print(read_motors)