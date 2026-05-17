def check_key (mess:str, key:str):
    """
    Raises an error if the key is shorter than the message .
    """
    assert len(key) >= len(mess)*8, " Keylen < messlen "

def encrypt(data:str, token:str):
    byte_encoded_data = data.encode("utf-8")

    data_len_bits = len(byte_encoded_data)*8

    int_encoded_data = int.from_bytes(byte_encoded_data)

    encrypted_mess_int = int_encoded_data ^ int(token, 2)

    # Transform the message back in bits, padding it to match the key.
    # Output a str

    encrypted_mess_str = format(encrypted_mess_int, f'0{data_len_bits}b')
    
    return(encrypted_mess_str)

# Now it is assumed the incoming message has the right length

def decrypt(ciphertext:str, token:str):
    
    data_len_bits = len(ciphertext)

    int_ciphertext = int(ciphertext, 2)

    decrypted_mess_int = int_ciphertext ^ int(token, 2)

    decrypted_word = decrypted_mess_int.to_bytes(data_len_bits // 8, 'big').decode('utf-8')

    return(decrypted_word)

token="111100011111000000010110010110"
data="INR"

check_key(data, token) 

encrypted = encrypt(data, token)

decrypted = decrypt(encrypted, token)

print(decrypted)
