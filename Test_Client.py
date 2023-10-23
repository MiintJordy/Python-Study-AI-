import socket
import threading
import os
# import time

# 접속할 server 정보
host = '10.10.21.110'
port = 30000


# client 소켓 생성
# client_socket = 소켓 모듈.소켓 class 이용하여 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

# 파일 이름
file_name = "abcdefg.mp4"
file_path = "C:\\Users\\lms110\\Downloads\\project\\send\\abcdefg.mp4"


# message 보내는 함수
def send_file():
    client_socket.send(file_name.encode('utf-8'))
    print(f"파일 이름: {file_name}")

    file_size = os.path.getsize(file_path)
    client_socket.send(str(file_size).encode('utf-8'))
    print(f"파일 크기: {file_size}")

    divide_size = 1024
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(divide_size)
            if not data:
                print("데이터 없음")
                break
            client_socket.send(data)
        print("파일 전송 완료")


# server message 수신
def receive_msg():
    while True:
        try:
            message = client_socket.recv(1024)
            print(message.decode('utf-8'))

        except Exception as e:
            print(f"서버 연결 종료: {e}")
            client_socket.close()
            break


send_file_thread = threading.Thread(target=send_file)
receive_msg_thread = threading.Thread(target=receive_msg)

send_file_thread.start()
receive_msg_thread.start()
