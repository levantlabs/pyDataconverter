from setuptools import setup

#Include in environment for testing:  pip install -e .
#This needs to be run in the directory before this

setup(
    name="pyDataconverter",
    version="0.1",
    packages=['pyDataconverter'],
    python_requires='>=3.7',
    install_requires=[
        # Add any dependencies your package needs here
    ],
    author="Manar El-Chammas",
    description="A Python data converter toolbox",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
    ],
)