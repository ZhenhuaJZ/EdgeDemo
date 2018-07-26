import socket
import cv2
import numpy as np
import re, ast

address = ('192.168.1.66', 8002)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(True)

def recvall(sock, bufsize):
    buf = b''
    while bufsize:
        temp_buf = sock.recv(bufsize)
        if not temp_buf: return None
        bufsize -= len(temp_buf)
        buf += temp_buf
    return buf

conn, addr = s.accept()

data_list = []

while True:
    # string list to int list
    data = recvall(conn,1)
    # data = numpy.fromstring(data, dtype='uint8', sep = " ")
    #print(data)
    if not data:
        break
    # print(data)
    data_list.append(data.decode("utf-8"))
    # data_list.append(data.decode("utf-8"))

def impose_to_img(pixels):
    img = np.zeros((224,224,3), np.uint8)
    img[:,:,:] = 255
    for i in range(len(pixels)):
        img[pixels[i][0], pixels[i][1]] = pixels[i][2:]
    return img

f_data = ''.join(data_list)

f_data = ast.literal_eval(f_data)
print(type(f_data))
print(f_data[0][1])

img = impose_to_img(f_data)
cv2.imshow("img",img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
s.close()

# address = ('127.0.0.1', 8002)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind(address)
# s.listen(True)
#
# def recvall(sock, count):
#     buf = b''
#     while count:
#         newbuf = sock.recv(count)
#         if not newbuf: return None
#         buf += newbuf
#         count -= len(newbuf)
#     return buf
#
# conn, addr = s.accept()
#
# while 1:
#     length = recvall(conn,16)
#     stringData = recvall(conn, int(length))
#     data = numpy.fromstring(stringData, dtype='uint8')
#     decimg=cv2.imdecode(data,1)
#     cv2.imshow('SERVER',decimg)
#     if cv2.waitKey(10) == 27:
#         break
#
# s.close()
# cv2.destroyAllWindows()
