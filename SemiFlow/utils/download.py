"""
@File : download.py
@Author: Dong Wang
@Date : 2020/6/9
"""
import hashlib
import sys
import requests
import os

requests.packages.urllib3.disable_warnings()


def md5_checksum(file_path):
    with open(file_path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    return checksum


def download(url, file, checksum):
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])
    if os.path.exists('./' + file):
        temp_size = os.path.getsize('./' + file)
        if md5_checksum('./' + file) == checksum:
            return './' + file
        else:
            print("Wrong checksum!")
    else:
        temp_size = 0
    print("Downloaded size:", temp_size, "Total size:", total_size)
    headers = {'Range': 'bytes=%d-' % temp_size}
    r = requests.get(url, stream=True, verify=False, headers=headers)

    with open('./' + file, "ab") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_size += len(chunk)
                f.write(chunk)
                f.flush()
                done = int(50 * temp_size / total_size)
                sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                sys.stdout.flush()
    print()
    # Todo: support customize path
    return './' + file


if __name__ == '__main__':
    url = "http://deeplearning.net/data/mnist/"
    file = "mnist.pkl.gz"
    checksum = "a02cd19f81d51c426d7ca14024243ce9"
    download(url + file, file, checksum)
