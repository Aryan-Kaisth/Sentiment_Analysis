from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.strip() for i in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="Sentiment analysis",
    version="0.1.0",
    author="Aryan",
    author_email="aryankaisthpvt@gmail.com",
    description="A Deep learning project using advanced LSTM that analyzes text data to determine sentiment as positive or negative",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.11",
)