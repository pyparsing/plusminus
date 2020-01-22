# setup.py
from setuptools import setup, find_packages
from plusminus import __version__ as pm_version

setup(
    name="plusminus",
    version=pm_version,
    packages=find_packages(),
    scripts=["say_hello.py"],

    install_requires=["pyparsing >= 2.4.6"],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # And include any *.msg files found in the "hello" package, too:
        "hello": ["*.msg"],
    },

    # metadata to display on PyPI
    author="Paul McGuire",
    author_email="me@example.com",
    description="This is an Example Package",
    keywords="hello world example examples",
    url="http://example.com/HelloWorld/",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://bugs.example.com/HelloWorld/",
        "Documentation": "https://docs.example.com/HelloWorld/",
        "Source Code": "https://code.example.com/HelloWorld/",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ]

    # could also include long_description, download_url, etc.
)