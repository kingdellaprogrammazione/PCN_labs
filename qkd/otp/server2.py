import socket

# Handmade functions to allow the framed communication

def send_framed(sock: socket.socket, payload: bytes):
    """Send a 4-byte big-endian length prefix followed by payload."""
    length_prefix = len(payload).to_bytes(4, 'big')
    sock.sendall(length_prefix + payload)

def recv_framed(sock):
    length_bytes = recv_exact(sock, 4)
    length = int.from_bytes(length_bytes, "big")
    payload = recv_exact(sock, length)
    return payload

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data

def get_key_from_file(key_id):
    with open(f'keys/{key_id}', 'rb') as f:
        key = f.read()
        return key

# Updated function to decrypt xoring in a ring-way
def decrypt(ciphertext: bytes, key: bytes) -> str:
    if len(key) == 0:
        raise ValueError("Key is empty")
    # XOR each byte of ciphertext with key (repeat key if needed)
    plaintext_bytes = bytes(c ^ key[i % len(key)] for i, c in enumerate(ciphertext))
    print("Message decrypted.")
    return plaintext_bytes.decode()

def handle_client(client_socket, key_id,encrypted_message):
    print(f"Received key id: {key_id}")
    #get key from file keys depending on the variable name key_id
    key = get_key_from_file(key_id)
    decrypted_data = decrypt(encrypted_message, key)
    response = f"Hello {decrypted_data}"
    print("Sending answer to client :", response)
    send_framed(client_socket, response.encode())

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.bind(('10.40.7.121', 8080))
    server_socket.bind(('127.0.0.1', 8080))

    server_socket.listen(1)

    print("Server is listening on port 8080...")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")

    key_id = recv_framed(client_socket).decode()
    encrypted_message = recv_framed(client_socket)

    handle_client(client_socket, key_id, encrypted_message)

    client_socket.close()
    server_socket.close()
main()




