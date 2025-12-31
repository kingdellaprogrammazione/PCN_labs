import socket

auth_token = 0b1010100010101010010010010110001011010001

def decrypt(ciphertext: bytes, key: bytes) -> str:
    decoded_word = int.from_bytes(ciphertext) ^ key
    old = int.from_bytes(ciphertext)
    return decoded_word.to_bytes(8).decode()

def handle_client(client_socket, data):
    print(f"Received: {data}")
    print(f"Encrypted Data: {data}")
    decrypted_data = decrypt(data, auth_token)
    print(f"Decrypted: {decrypted_data}")

    response = f"Hello {decrypted_data}"
    client_socket.send(response.encode())



def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('10.40.7.121', 8080))
    # server_socket.bind(('127.0.0.1', 8080))

    server_socket.listen(1)

    print("Server is listening on port 8080...")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")

    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        handle_client(client_socket, data)


    client_socket.close()
    server_socket.close()
main()
