from setuptools import setup, find_packages

setup(
    name="VAE",
    version="0.1.0",
    author="Prashant Gavit",
    author_email="prashantgavit115@gmail.com",
    description="",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prashantgavit/variational-autoencoder",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",  # Example additional classifier
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.11',
)  