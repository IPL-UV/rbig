import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_rbig",
    version="0.0.1",
    author="J. Emmanuel Johnson",
    author_email="jemanjohnson34@gmail.com",
    description="A scikit-learn compatible package that Gaussianizes multidimensional data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jejjohnson/rbig",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
