import socket
import time


class SimpleSocket:
    sep = ";"

    def __init__(self, address="127.0.0.1", port=8000):
        self.sock = None
        self.address = address
        self.port = port
        self.last_delay = -1  # ms

    def connect(self):
        self.sock = socket.socket()
        self.sock.connect((self.address, self.port))

    def close(self):
        self.sock.close()

    def send(self, cmd, *args):
        parts = [str(p) for p in [cmd, *args]]
        msg = self.sep.join(parts) + "\n"
        cur_time = time.time()

        self.sock.send(msg.encode())

        resp = self.sock.recv(256)
        self.last_delay = (time.time() - cur_time) * 1000 / 2

        return resp
