import socket

def get_key_from_file(key_id):
    with open(f'keys/{key_id}.key', 'rb') as f:
        print(f.read())
        return f.read()

def decrypt(ciphertext: bytes, key: bytes) -> str:
    decoded_word = int.from_bytes(ciphertext) ^ int.from_bytes(key)
    old = int.from_bytes(ciphertext)
    return decoded_word.to_bytes(8).decode()

def handle_client(client_socket, key_id,encrypted_message):
    print(f"Received: {key_id}")
    #get key from file keys depending on the variable name key_id
    key = get_key_from_file(key_id)
    decrypted_data = decrypt(encrypted_message, key)
    response = f"Hello {decrypted_data}"
    client_socket.send(response.encode())



def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('10.40.7.121', 8080))
    server_socket.listen(1)

    print("Server is listening on port 8080...")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")

    while True:
        data = client_socket.recv(1024)
        [key_id, encrypted_message] = data.decode().split(',')
        if not data:
            break
        print(key_id,encrypted_message)

        handle_client(client_socket, key_id,encrypted_message.encode())


    client_socket.close()
    server_socket.close()
main()
