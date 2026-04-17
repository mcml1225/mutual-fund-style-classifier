from setuptools import setup, find_packages

setup(
    name="mutual-fund-style-classifier",
    version="1.0.0",
    author="Your Name",
    description="Unsupervised ML for classifying mutual funds into Morningstar Style Box",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'yfinance>=0.2.0',
        'streamlit>=1.25.0',
        'plotly>=5.15.0',
    ],
)