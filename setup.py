#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

# About dict to store version and package info
about = dict()
with open('peartree/__version__.py', 'r', encoding='utf-8') as f:
    exec(f.read(), about)

requirements = [
    'fiona>=1.6.1',
    'networkx>=2.0',
    'osmnx==0.6',
    'partridge==0.3.0'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    name='peartree',
    version=about['__version__'],
    description=('Peartree is a library for '
                 'converting GTFS to directed graphs.'),
    long_description=readme,
    author='Kuan Butts',
    author_email='kuanbutts@gmail.com',
    url='https://github.com/kuanb/peartree',
    packages=find_packages(include=['peartree']),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    keywords='peartree',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
