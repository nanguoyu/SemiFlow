"""
@File : setup.py
@Author: Dong Wang
@Date : 2020/4/26
"""

import setuptools

setuptools.setup(
    name='SemiFlow',
    version='1.1.3',
    description='SemiFlow is a deep learning framework with automatic differentiation and automatic shape inference, '
                'developing from Numpy.',
    url='https://github.com/nanguoyu/SemiFlow',
    author='Dong Wang (nanguoyu)',
    author_email='dongwang@wangdongdong.wang',
    maintainer='Dong Wang (nanguoyu)',
    maintainer_email='dongwang@wangdongdong.wang',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'onnx'],
    setup_requires=["numpy>=1.14.0"],
    python_requires='>=3.8'
)
