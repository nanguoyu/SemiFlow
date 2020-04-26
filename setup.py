"""
@File : setup.py
@Author: Dong Wang
@Date : 2020/4/26
"""

import setuptools

setuptools.setup(
    name='SemiFlow',
    version='0.0.2',
    description='Implement a deep learning framework from zero(strictly Numpy).',
    url='https://github.com/nanguoyu/SemiFlow',
    author='Dong Wang (nanguoyu)',
    author_email='admin@nanguoyu.com',
    maintainer='Dong Wang (nanguoyu)',
    maintainer_email='admin@nanguoyu.com',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib'],
    setup_requires=["numpy>=1.14.0"],
    python_requires='>=3.6'
)
