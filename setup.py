import setuptools

setuptools.setup(
    name='pycrystalfield',
    version='2.3.3',    
    description='Code to calculate the crystal field Hamiltonian of magnetic ions.',
    url='https://github.com/asche1/PyCrystalField',
    author='Allen Scheie',
    author_email='',
    license='GNU GPL',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['matplotlib',
                      'scipy',
                      'numba', 'numpy'   
                      ],
    package_data={'pcf_lib':['*',]}
)