from setuptools import setup, find_packages

setup(
    name="core-road-extraction",
    version="1.0.0",
    description="Canopy-Occluded Road Extraction from Satellite Imagery using D-LinkNet",
    author="Hari",
    packages=find_packages(exclude=["data*", "outputs*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0",
        "albumentations==1.4.18",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
