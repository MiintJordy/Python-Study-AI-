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


def read_data(client_file):
    while True:
        try:
            # 파일 이름
            file_name = client_file.recv(1024).decode('utf-8')
            # 파일 저장 경로
            file_path = "C:\\Users\\lms110\\Downloads\\project\\receive\\"
            # 저장할 파일 경로 + 파일 이름
            file_info = file_path + file_name

            # 파일 크기
            file_size = int(client_file.recv(1024).decode('utf-8'))
            print(f"파일 이름: {file_name},  파일 크기: {file_size} bytes")

            # 데이터 파일 저장
            with open(file_info, 'wb') as file:
                while file_size > 0:
                    data = client_file.recv(1024)
                    if not data:
                        print("데이터 없음")
                        break
                    file.write(data)
                    file_size -= len(data)
                    print(file_size)

                print(f"{file_name} 파일 수신 완료")

        except Exception as e:
            print(f"에러 발생: {e}")
            break
    client_file.close()
    client_socket_list.remove(client_file)


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
