import socket
import threading

# 서버 정보
host = '10.10.21.110'
port = 25000

# 서버 socket 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

# client 정보를 담을 리스트
client_socket_list = []


def read_data(client_msg):
    while True:
        try:
            message = client_msg.recv(1024)

            # 수신한 메시지 전송
            for client in client_socket_list:
                client.send(message)

            print(message.decode('utf-8'))

        except Exception as e:
            print(f"에러 발생: {e}")
            break
    client_socket.close()
    client_socket_list.remove(client_socket)


while True:
    client_socket, client_address = server_socket.accept()
    client_socket_list.append(client_socket)
    client_accept = threading.Thread(target=read_data, args=(client_socket,))
    client_accept.start()
