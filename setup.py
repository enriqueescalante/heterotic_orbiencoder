from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'Autoencoder for dimensional reduction of effective string model parameters'


# Setup

setup(
    name="AEST",
    version=VERSION,
    description=DESCRIPTION,
    author="Enrique Escalante-Notario&Ignacio Portillo-Castillo& Saúl Ramos-Sánchez",
    author_email="enriquescalante@gmail.com",
    url="http://github.com/enriqueescalante",
    install_requires=['jupyter','numpy', 'torch', 'scikit-learn', 'pandas', 'matplotlib'],
    scripts=[]
)