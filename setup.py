#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    # TODO: put package requirements here
    # for python 2.7 pathlib
    # scikit-image
    # opencv
    # tqdm
    # https://github.com/openearth/bmi-python/archive/master.zip (make a package)
    # cmocean
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='sandbox_fm',
    version='0.1.0',
    description="Sandbox combined with a Delft3D Flexbile Mesh simulation",
    long_description=readme + '\n\n' + history,
    author="Fedor Baart",
    author_email='fedor.baart@deltares.nl',
    url='https://github.com/openearth/sandbox_fm',
    packages=[
        'sandbox_fm',
    ],
    package_dir={'sandbox_fm':
                 'sandbox_fm'},
    package_data={
        'sandbox_fm': ['data/*.png']
    },
    entry_points={
        'console_scripts': [
            'sandbox-fm=sandbox_fm.cli:cli'
        ]
    },
    # install all scripts
    scripts=[
        os.path.join('scripts', path)
        for path
        in os.listdir('scripts')
    ],
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='sandbox_fm',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
