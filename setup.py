#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='Word Sense Disambiguation for Neural Language Models',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='hhttps://github.com/jpelhaW/wsd',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

