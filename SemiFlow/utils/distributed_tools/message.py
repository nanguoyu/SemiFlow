"""
@File : message.py
@Author: Dong Wang
@Date : 7/18/2021
"""
import json
import struct


class Message:
    def __init__(self):

        self.json_header = None

    def read_msg_header(self, conn):
        header_len_b = conn.recv(4)
        if not header_len_b:
            return None
        header_len = struct.unpack('i', header_len_b)[0]
        return header_len

    def read_msg(self, conn, header_len):
        header_b = conn.recv(header_len)
        if not header_b:
            return None
        header_str = header_b.decode()
        self.json_header = json.loads(header_str)
        action = self.json_header.get('action')
        if action == 'update model':
            data_len = self.json_header['data_size']
            data_name = self.json_header['data_name']
            self.json_header = None
            return 'update model', data_len, data_name
        elif action == 'send model':
            data_len = self.json_header['data_size']
            data_name = self.json_header['data_name']
            self.json_header = None
            return 'send model', data_len, data_name
        elif action == 'get model':
            self.json_header = None
            return 'get model', 0, None
        else:
            self.json_header = None
            return None, 0, None

    def read_data(self, conn, data_len, data_name):
        print(f'Try to load {data_name}')
        data = b''
        while data_len > 0:
            content = conn.recv(1024)
            data += content
            data_len -= len(content)

        return data

    def send_data(self, conn, data, data_name, action):
        data_size = len(data)
        dic = {'data_name': data_name, 'data_size': data_size, 'action': action}
        print(dic)
        x = len(json.dumps(dic).encode())
        msg_len = struct.pack('i', x)
        conn.send(msg_len)
        conn.send(json.dumps(dic).encode())
        while data_size > 0:
            if data_size >= 1024:
                content = data[:1024]
                conn.send(content)
                data = data[1024:]
                data_size -= 1024
            else:
                content = data[:data_size]
                conn.send(content)
                data_size -= data_size
