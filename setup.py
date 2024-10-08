#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    setup.py
    ~~~~~~~~

    Demonstrates Speech to Speech with Groq & OpenAI APIs

    :copyright: (c) 2024 by Aryan Mistry, Harish Kashyap.
    :license: see LICENSE for more details.
"""

import codecs
import os
import re
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """Taken from pypa pip setup.py:
    intentionally *not* adding an encoding option to open, See:
       https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    """
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='s2s',
    version=find_version("s2s", "__init__.py"),
    description='Demonstrates Speech to Speech with Groq & OpenAI APIs',
    long_description=read('README.md'),
    classifiers=[
        ],
    author='Aryan Mistry, Harish Kashyap',
    author_email='aryan@pandita.ai, harish@pandita.ai',
    url='',
    packages=[
        's2s'
        ],
    platforms='any',
    license='LICENSE',
    install_requires=[
        'virtualenv>=1.11.6',
        'pep8>=1.5.7',
        'pyflakes>=0.8.1',
        ]
)
