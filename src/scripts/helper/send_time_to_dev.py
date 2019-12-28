import socket
import time

# noinspection PyUnresolvedReferences
import pathmagic

# adb forward tcp:8000 tcp:9000
from dataset_tools.simple_socket import SimpleSocket


def send_time(sock, exp_delay_ms):
    time_ms = int(round(time.time() * 1000))
    resp = sock.send("time", time_ms, exp_delay_ms)

    if resp.decode() != "ok":
        print("Failure")
        exit(1)

    return sock.last_delay


def sync_dev_time(sock):
    last_delay = 99999
    delay_diff = 99999

    while delay_diff != 0:
        cur_delay = send_time(sock, last_delay)
        print("RTT/2: {:.3f}ms".format(cur_delay))
        cur_delay = int(round(cur_delay))
        delay_diff = cur_delay - last_delay
        last_delay = cur_delay

    print("Done syncing time")


def main():
    sock = SimpleSocket(address="127.0.0.1", port=8000)
    sock.connect()

    sync_dev_time(sock)

    sock.close()


if __name__ == "__main__":
    main()
