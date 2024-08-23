from setuptools import setup, find_packages

setup(
    name="nanotrack",
    version="0.1.0",
    author="RAGUL T , KARTHICK RAJA E",
    author_email="tragulragul@gmail.com , e.karthickraja2004@gmail.com",
    description="A lightweight object detection and tracking package",
    url="https://github.com/yourusername/nanotrack",  # Update with your actual GitHub repo URL
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "opencv-python>=4.5.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "scipy>=1.5.0",
    ],
)
