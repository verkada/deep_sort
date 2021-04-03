#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='deep_sort',
      version='1.0.0',
      description='Deep sort for object tracking.',
      url='git@github.com:williammc/deep_sort.git',
      author='ZQPei',
      author_email='',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE',
    )
