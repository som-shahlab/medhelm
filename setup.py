from setuptools import setup, find_packages

setup(
    name='medhelm',
    version='0.1.0',    
    description='Package used to analyze results for MedHELM',
    url='https://github.com/som-shahlab/medhelm',
    author='Miguel Fuentes',
    author_email='migufuen@stanford.edu',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'crfm-helm',
    ]
)
