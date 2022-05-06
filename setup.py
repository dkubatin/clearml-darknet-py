from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='clearml-darknet-py',
    version='0.1.3',
    description='Python package for training neural networks through the Darknet framework in ClearML',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dkubatin/clearml-darknet-py',
    author='Denis Kubatin',
    author_email='kubatin.denis@gmail.com',
    license='MIT',
    keywords=["darknet", "clearml"],
    platforms="any",
    packages=['clearml_darknet'],
    python_requires=">=3.4",
    install_requires=['clearml==1.3.2'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ]
)
