from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='clearml-darknet-py',
    version='0.2.0-alpha.2',
    description='The library reproduces the reproduction of neural networks on the Darknet framework in ClearML',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dkubatin/clearml-darknet-py',
    author='Denis Kubatin',
    author_email='kubatin.denis@gmail.com',
    license='MIT',
    keywords=["darknet", "clearml", "machine-learning", "deep-learning", "neural-network"],
    platforms="any",
    packages=['clearml_darknet'],
    python_requires=">=3.6",
    install_requires=['clearml==1.3.2'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Version Control',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ]
)
