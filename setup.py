from setuptools import setup, find_packages

setup(
    name="mlphase",
    version="0.0.1",
    description="Accelerating Multicomponent Phase-Coexistence Calculations with Machine Learning",
    author="Satyen Dhamankar and Shengli (Bruce) Jiang",
    author_email="sj0161@princeton.com",
    url="https://github.com/webbtheosim/ml-ternary-phase",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.4",
        "matplotlib==3.4.3",
        "torch==1.13.1",
        "proplot==0.9.7",
        "scikit-learn==1.2.2",
        "autograd==1.6.2",
    ],
)
