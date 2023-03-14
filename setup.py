import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
   
def dependencies_from_file(file_path):
    required = []
    with open(file_path) as f:
        for l in f.readlines():
            l_c = l.strip()
            # get not empty lines and ones that do not start with python
            # comment "#" (preceded by any number of white spaces)
            if l_c and not l_c.startswith('#'):
                required.append(l_c)
    return required

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
    install_requires=dependencies_from_file('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
