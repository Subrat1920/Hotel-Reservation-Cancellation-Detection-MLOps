from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirements = []
    try:
        with open("requirements.txt") as f:
            for line in f:
                req = line.strip()
                if req and req != "-e .":
                    requirements.append(req)
    except FileNotFoundError:
        print("requirements.txt not found!")
    return requirements

setup(
    name="Hotel-Reservation-System-Cancellation-Detection",
    version="0.0.1",
    author="Subrat Mishra",
    author_email="3subratmishra1sep@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
