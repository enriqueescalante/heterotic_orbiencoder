from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'Autoencoder for dimensional reduction of effective string model parameters'


# Setup

setup(
    name="heterotic_orbiencoder",
    version=VERSION,
    description=DESCRIPTION,
    author="Enrique Escalante-Notario, Ignacio Portillo-Castillo,  Saúl Ramos-Sánchez",
    author_email="enriquescalante@gmail.com",
    url="https://github.com/enriqueescalante/heterotic_orbiencoder",
    install_requires=['jupyter','numpy', 'torch', 'scikit-learn', 'pandas', 'matplotlib'],
    scripts=[]
)