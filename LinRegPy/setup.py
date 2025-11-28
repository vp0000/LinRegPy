from setuptools import setup, find_packages

setup(
    name='LinRegPy',  # The package name for imports (e.g., from LinRegPy.base_model import ...)
    version='0.1.0',
    description='A NumPy-based Linear Regression Library with Diagnostics (Hobby Project)',
    author='Your Name',
    
    # Automatically discovers the PyStatLin/ source directory
    packages=find_packages(), 
    
    # List all necessary external libraries
    install_requires=[
        'numpy>=1.20',
        'pandas>=1.3',
        'scipy>=1.7',
    ],
    
    python_requires='>=3.8',

)
