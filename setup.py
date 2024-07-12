from setuptools import setup, find_packages

setup(
    name='LOCAETA_AQ',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'geopandas',
        'requests',
        'pyproj',
        'shapely',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'nei-csv-to-shapefile=LOCAETA_AQ.NEI_csv_to_shapefile:main',
            'incorporate-ccs-to-nei=LOCAETA_AQ.Incorporate_CCS_to_NEI:main',
        ],
    },
    author='Yunha Lee',
    author_email='yunha.lee@carbonsolutionsllc.com',
    description='Local Climate and Air Emissions Tracking Atlas (LOCAETA) - Air Quality',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yunh-lee-CS/LOCAETA_AQ',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)