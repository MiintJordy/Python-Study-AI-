import socket
import threading
# import time

# 접속할 server 정보
host = '10.10.21.110'
port = 30000


# client 소켓 생성
# client_socket = 소켓 모듈.소켓 class 이용하여 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))


# message 보내는 함수
def send_msg():
    while True:
        message = input()
        to_send_server = 'msg' + '/!@#/' + message
        client_socket.send(to_send_server.encode('utf-8'))


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


send_msg_thread = threading.Thread(target=send_msg)
receive_msg_thread = threading.Thread(target=receive_msg)

send_msg_thread.start()
receive_msg_thread.start()
