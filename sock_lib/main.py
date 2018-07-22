import os
import socket

PORT = 8080
HOST = 'localhost'
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind((HOST,PORT))
socket.listen(1)
conn, addr = socket.accept()


