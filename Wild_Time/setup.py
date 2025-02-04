import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wildtime",
    version='1.0.6',
    author="wildtime team",
    author_email="wilds@cs.stanford.edu",
    url="https://wilds.stanford.edu",
    description="WILDS distribution shift benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'gdown==4.5.1',
        'ipdb==0.13.9',
        'lightly==1.2.27',
        'matplotlib==3.5.0',
        'numpy==1.19.5',
        'omegaconf==2.0.6',
        'pandas>=1.1.3',
        'Pillow==9.2.0',
        'pytorch_lightning==1.3.6',
        'scikit_learn==1.1.2',
        'PyTDC==0.3.7',
        # 'torch==1.12.1',
        'pytorch_tabular==0.7.0',
        'torchcontrib==0.0.2',
        'torchvision>=0.8.1',
        'torchmetrics==0.6.0',
        'transformers==4.21.1',
        'wilds==2.0.0'
    ],
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
