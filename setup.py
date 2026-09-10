from setuptools import setup, find_packages

setup(
    name='genbop',
    version='0.1.0',
    author='Ikuma Kohata',
    author_email='ikuma0526@g.ecc.u-tokyo.ac.jp',
    description='Generalized Bond-Order Potential',
    packages=find_packages(),
    install_requires=[
        'numpy==2.5.1',
        'ase==3.29.0',
        'torch',
        'torch_geometric',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)