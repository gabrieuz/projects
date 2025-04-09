from setuptools import setup, find_packages

setup(
    name="cnn-fashion-mnist-comparison",
    version="0.1.0",
    author="Gabriel Martins",
    author_email="gabrieldlm@outlook.com",
    description="A project for comparing various CNN architectures on the FashionMNIST dataset.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.4.0",
        "pandas>=1.2.0",
        "plotly>=5.0.0",
        "numpy>=1.19.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    project_urls={
        'GitHub': 'https://github.com/gabrieuz',
        'LinkedIn': 'https://linkedin.com/in/gabrieldlm',
    },
)
