from distutils.core import setup

setup(
    name='tensorflow3d',
    packages=['tensorflow3d'],
    version='0.1',
    license='MIT',
    description='An extension to tensorflow framework for 3D applications',
    author='Mohammad Asim',
    author_email='asim.98.12.26@gmail.com',
    url='https://github.com/mohammadasim98/tensorflow3d.git',
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
    keywords=['tensorflow3d', '3D Deep Learning'],
    install_requires=[
        'tensorflow',
        'open3d'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
