  
from setuptools import setup

def parse_requirements(filename):
	lines = (line.strip() for line in open(filename))
	return [line for line in lines if line and not line.startswith("#")]

setup(
    name='dall-e-tok',
    version='0.1',
    description='Huggingface package for the discrete VAE usded for DALL-E.',
    url='https://github.com/jinmang2/DALL-E-Tokenizer.git',
    author='MyungHoon Jin',
    author_email='jinmang2@gmail.com',
    packages=['dall_e_tok'],
    install_requires=parse_requirements('requirements.txt'),
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
