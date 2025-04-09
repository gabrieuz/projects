from setuptools import setup, find_packages

setup(
    name="pytorch_introduction",
    version="0.1.0",
    description="A structured project for learning PyTorch basics",
    author="Gabriel Martins",
    author_email="gabrieldlm@outlook.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        'GitHub': 'https://github.com/gabrieuz',
        'LinkedIn': 'https://linkedin.com/in/gabrieldlm',
    },
)
