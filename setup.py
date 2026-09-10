from setuptools import setup, find_packages

setup(
    name='genbop',
    version='0.1.0',
    author='Ikuma Kohata',
    author_email='kohata@photon.t.u-tokyo.ac.jp',
    description='Generalized Bond-Order Potential',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ase',
        'torch',
        'torch_geometric',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)