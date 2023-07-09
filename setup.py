from setuptools import setup, find_packages

setup(
    name="pytorchts",
    version="0.6.0",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.8.0",
        "gluonts==0.10.0",
        "holidays",
        "numpy~=1.16",
        "pandas~=1.1",
        "scipy",
        "tqdm",
        "matplotlib",
        "tensorboard",
    ],
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
