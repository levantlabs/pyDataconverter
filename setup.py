from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pyDataconverter",
    version="0.03",
    packages=find_packages(exclude=['tests']),
    # PEP 561: ship the py.typed marker so type-checker consumers (mypy,
    # pyright, etc.) pick up the inline annotations.
    package_data={'pyDataconverter': ['py.typed']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    author="Levant Labs",
    description="A Python toolbox for modeling and analyzing data converters",
    long_description=long_description,
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
