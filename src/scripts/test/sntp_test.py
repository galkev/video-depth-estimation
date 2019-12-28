import datetime
import struct
import time
from _socket import socket, AF_INET, SOCK_DGRAM
from contextlib import closing

import ntplib


class NtpTime:
    def __init__(self, server="time.google.com"):
        self.client = ntplib.NTPClient()
        #self.server = server
        self.server = 'pool.ntp.org'

        self.offset = 0

        self.sync()

    def sync(self):
        self.offset = self.client.request(self.server).offset
        print(self.offset)

    def now(self):
        return time.time() + self.offset

    def now_datetime(self):
        return datetime.datetime.fromtimestamp(self.now())


def ntp_time_f(host="pool.ntp.org", port=123):
    NTP_PACKET_FORMAT = "!12I"
    NTP_DELTA = 2208988800  # 1970-01-01 00:00:00
    NTP_QUERY = '\x1b' + 47 * '\0'

    with closing(socket(AF_INET, SOCK_DGRAM)) as s:
        s.sendto(NTP_QUERY, (host, port))
        msg, address = s.recvfrom(1024)
    unpacked = struct.unpack(NTP_PACKET_FORMAT,
                             msg[0:struct.calcsize(NTP_PACKET_FORMAT)])
    return unpacked[10] + float(unpacked[11]) / 2 ** 32 - NTP_DELTA


if __name__ == '__main__':
    ntp_time = NtpTime()

    timestamp_fmt = "%H:%M:%S:%f"

    print(ntp_time_f())
    exit(0)
    while True:
        ntp_time.sync()
        timestamp = ntp_time.now_datetime()
        #print(timestamp.strftime(timestamp_fmt))
        time.sleep(0.1)
        #break

