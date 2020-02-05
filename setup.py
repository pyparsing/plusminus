# setup.py
from setuptools import setup, find_packages
from plusminus import __version__ as pm_version

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="plusminus",
    version=pm_version,
    packages=find_packages(),

    install_requires=["pyparsing>=2.4.6"],

    # metadata to display on PyPI
    author="Paul McGuire",
    author_email="ptmcg@austin.rr.com",
    description="""
        +/- plusminus is a module that builds on the pyparsing infixNotation helper method to build easy-to-code and 
        easy-to-use parsers for parsing and evaluating infix arithmetic expressions. plusminus's ArithmeticParser 
        class includes separate parse and evaluate methods, handling operator precedence, override with parentheses, 
        presence or absence of whitespace, built-in functions, and pre-defined and user-defined variables, functions, 
        and operators.
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="python infix notation arithmetic safe eval",
    url="https://github.com/pyparsing/plusminus",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/pyparsing/plusminus/issues",
        "Documentation": "https://github.com/pyparsing/plusminus",
        "Source Code": "https://github.com/pyparsing/plusminus",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: User Interfaces",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Utilities",
    ],
    python_requires='>=3.5.2',
)
