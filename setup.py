import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scmer-lshh125",
    version="v0.1.0-alpha",
    author="Shaoheng Liang",
    author_email="",
    description="Marker selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
