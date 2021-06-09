from setuptools import setup

setup(
    name='pycurve',
    version='0.1.0',
    description='Analyze the curvature from a figure',
    url='',
    author='Jianzhe Gu',
    author_email='',
    license='BSD 2-clause',
    packages=['pycurve'],
    install_requires=['opencv-python>=4.2',
                      'numpy>=1.14.5',
                      'matplotlib>=3.3.4',
                      'scikit-image>=0.16.2'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)