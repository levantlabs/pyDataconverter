from setuptools import setup

setup(
    name="pyDataconverter",
    version="0.02",
    packages=['pyDataconverter'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    author="Levant Labs",
    description="A Python toolbox for modeling and analyzing data converters",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/levantlabs/pyDataconverter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)"
    ]
)