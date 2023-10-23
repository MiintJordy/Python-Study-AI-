import socket
import threading

# 서버 정보
host = '10.10.21.110'
port = 30000

# 서버 socket 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

# client 정보를 담을 리스트
client_socket_list = []

# 파일 정보 담기
total_file_size = 0
file_name = ''
save_path = ''


def read_data(client_msg):
    while True:
        try:
            read_message = client_msg.recv(1024)
            decode_message = read_message.decode('utf-8')
            split_message = decode_message.split('/!@#/')

            if split_message[0] == 'msg':
                # 수신한 메시지 전송
                for client in client_socket_list:
                    client.send(split_message[1].encode('utf-8'))
                print(split_message[1])

        except Exception as e:
            print(f"에러 발생: {e}")
            break
    client_socket.close()
    client_socket_list.remove(client_socket)


def send_msg():
    while True:
        send_message = input()
        for client in client_socket_list:
            client.send(send_message.encode('utf-8'))


while True:
    client_socket, client_address = server_socket.accept()
    client_socket_list.append(client_socket)

    print(f"클라이언트가 {client_address}에서 연결되었습니다.")

    client_accept = threading.Thread(target=read_data, args=(client_socket,))
    to_client_msg = threading.Thread(target=send_msg)

    client_accept.start()
    to_client_msg.start()
