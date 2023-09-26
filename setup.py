from setuptools import setup

setup(
    name='thecov',
    version='0.1.4',    
    description='A python package to compute analytic covariance matrices based on Wadekar & Scoccimarro 2019 (arXiv:1910.02914).',
    url='https://github.com/cosmodesi/thecov',
    author='Otavio Alves',
    author_email='oalves@umich.edu',
    license='',
    packages=['thecov'],
    install_requires=['numpy', 'scipy', 'tqdm', 'pypower', 'mockfactory'],
    classifiers=[]
)
