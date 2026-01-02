#!/usr/bin/env python3
"""
Client that:
 1) sends a label (filename in keys/ directory)
 2) sends the encrypted message (XOR with key read from keys/<label>)

Framing: each message is prefixed by a 4-byte big-endian length.
Key file parsing supports:
 - raw bytes (binary file)
 - ASCII text (used literally)
 - hex (if file looks like hex)
 - base64 (if file looks like base64)
"""

import os
import socket
import base64
import binascii

# HOST = '10.40.7.121'   # change to your server
HOST = '127.0.0.1'   # change to your server

PORT = 8080        # change to your server port
KEYS_DIR = 'keys'    # directory containing key files
MESSAGE = "Ball_"  # the plaintext message to encrypt and send


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

def read_key_file(label: str) -> bytes:
    """Load key data from keys/<label>. Auto-detect format (raw, hex, base64, ascii)."""
    path = os.path.join(KEYS_DIR, label)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Key file not found: {path}")

    # open in binary to preserve bytes for raw keys
    with open(path, 'rb') as f:
        content = f.read()

    # Try heuristics:
    # 1) If file contains non-printable bytes -> assume raw binary key
    if any(b < 32 or b > 126 for b in content if b not in (9,10,13)):
        return content

    # decode as UTF-8 text for further heuristics
    text = content.decode('utf-8', errors='ignore').strip()

    # 2) looks like hex? (only hex chars and even length)
    hexchars = set("0123456789abcdefABCDEF")
    if len(text) >= 2 and all(c in hexchars for c in text.replace('\n','').replace('\r','')):
        s = ''.join(text.split())
        if len(s) % 2 == 0:
            try:
                return binascii.unhexlify(s)
            except Exception:
                pass

    # 3) looks like base64? try decode
    try:
        b = base64.b64decode(text, validate=True)
        # additional check: base64 decoding returning something plausible
        if len(b) > 0:
            return b
    except Exception:
        pass

    # 4) otherwise treat as UTF-8 text and use its bytes
    return text.encode('utf-8')

def get_key_labels(keys_dir='keys'):
    """
    Reads the list of key files (labels) from the given directory.
    Each filename represents one key label.
    
    Returns:
        list[str]: list of label names (filenames without directories)
    """
    if not os.path.exists(keys_dir):
        raise FileNotFoundError(f"Keys directory not found: {keys_dir}")

    labels = []
    for entry in os.listdir(keys_dir):
        path = os.path.join(keys_dir, entry)
        # include only files (not directories or hidden files)
        if os.path.isfile(path) and not entry.startswith('.'):
            labels.append(entry)

    return labels

def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR data with key repeating key if necessary."""
    if len(key) == 0:
        raise ValueError("Key is empty")
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

def main():
    # 1) load key
    label=get_key_labels()[0] 
    try:
        key_bytes = read_key_file(label)
    except Exception as e:
        print("Error loading key:", e)
        return

    print(f"Loaded key of {len(key_bytes)} bytes from '{os.path.join(KEYS_DIR, label)}'")

    # 2) prepare label message (UTF-8)
     # pick the first label for this example
    label_msg = label.encode('utf-8')

    # 3) prepare encrypted message
    plaintext = MESSAGE.encode('utf-8')
    encrypted = xor_bytes(plaintext, key_bytes)

    print('Encrypted message:', encrypted)
    # --- open socket and send both messages ---
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            # send label first
            send_framed(s, label_msg)
            print(f"Sent label ({len(label_msg)} bytes): {label!r}")

            # send encrypted message second
            send_framed(s, encrypted)
            print(f"Sent encrypted message ({len(encrypted)} bytes)")

            # optionally read server reply
            # read 4-byte length then payload (if server uses same framing)
            s.settimeout(5.0)
            try:
                answer = recv_framed(s).decode()

                print("Server reply :", answer)
            except socket.timeout:
                print("No response received (socket timeout).")

    except Exception as e:
        print("Socket error:", e)

if __name__ == "__main__":
    main()
