import socket

HOST = '10.40.7.121'  # Server address
# HOST = '127.0.0.1'  # Server address
PORT = 8080       # The same port as used by the server

# This is the symmetric random key obtained from the QKD synthesis
auth_token = 0b1010100010101010010010010110001011010001

# This is what we want to encrypt
data = "Ball_"

def encrypt(plaintext: str, key: bytes) -> bytes:

    # Transform the plaintext in bits
    plain_to_bytes = plaintext.encode()
    # Transform the plaintext in a integer ready to be xored
    encrypted_word = int.from_bytes(plain_to_bytes) ^ key
    # Encode it back into bytes, ready to be sent
    return encrypted_word.to_bytes(8)



def main():
    
    print("Trying connection to server...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    print("Plaintext:",data)
    client_socket.sendall(encrypt(data, auth_token))

    client_socket.settimeout(10.0)  # seconds

    print('Waiting for an answer from server...')
    received_data = client_socket.recv(1024)

    print('Received from server:', received_data.decode())
    client_socket.close()

main()