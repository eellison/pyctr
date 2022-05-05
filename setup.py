from setuptools import setup, find_packages

__version__ = 0.1

setup(
    name='pyctr',
    version=__version__,
    description=
    'Virtualization and source code transformation engine for Python',
    author='Pyctr team',
    author_email='',  # TODO Add author email,
    packages=find_packages(),
    install_requires=['gast<=0.5', 'astor', 'absl-py', 'termcolor'],
    url='https://github.com/google/pyctr',
    license='Apache-2.0',
)
