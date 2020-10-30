#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='1.0.0',
    description='Word Sense Disambiguation for Neural Language Models',
    author='', # Upon acceptance
    author_email='', # Upon acceptance
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='hhttps://github.com/jpelhaW/wsd',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

