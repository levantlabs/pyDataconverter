from setuptools import setup


setup(
    name="pyDataconverter",
    version="0.01",
    packages=['pyDataconverter'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    author="Manar El-Chammas",
    description="A Python toolbox for modeling and analyzing data converters",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyDataconverter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::  GPL-3.0 license ",
        "Operating System :: OS Independent",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
    ],
)