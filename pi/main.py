import cv2
from PIL import Image
import os
import time
import socket
import json
import struct
import numpy as np

ONE_SECOND = 1
RECOG_TIME = 10
RECOG_MIN_PERCENT = 0.5
PORT = 8080
HOST = 'localhost'

save_dir = "./save/"


def save_ndarray_as_image_of_name(filename, nparray):
    im = Image.fromarray(nparray)
    im.save(save_dir + filename)

def get_time_date():
    return  "-".join([str(t) for t in time.localtime(time.time())[0:5]])

def take_pic(cap):
    _, frame = cap.read()
    return frame

def motion_detected():
    return True

# def send_files():
#     [ for f in os.listdir(save_dir)]

if __name__ == "__main__":
    # make sure save dir exist
    try:
        os.mkdir(save_dir)
    except Exception as e:
        print("Directory \"save\" exists, proceed.")

    # initializing camera and wait for it to warm up
    cap = cv2.VideoCapture(0)
    time.sleep(ONE_SECOND)
    print("Camera have warmed up! Proceeding...")

    # start socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Socket initiated")

    while(1):
        if motion_detected():
            for i in range(3):
                take_pic(cap)


            time_start = time.time()
            time_elapsed = time.time() - time_start
            while time_elapsed < RECOG_TIME:
                time_elapsed = time.time() - time_start
                data = json.dumps(take_pic(cap).tolist())
                try:
                    json.loads(data)
                except Exception as e:
                    print(e)
                    exit(-1)
                sock.sendall(struct.pack("I", len(data)))
                sock.sendall(data.encode())

                l = sock.recv(4)
                l = int(struct.unpack("I", l)[0])
                result = json.loads(sock.recv(l).decode())
                print(result)
            end_message = "Done"
            sock.sendall(struct.pack("I", len(end_message)))
            sock.sendall(end_message.encode())
            break

