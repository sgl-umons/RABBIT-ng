from os import path
from setuptools import setup, find_packages
from codecs import open # To use a consistent encoding


__package__ = 'rabbit'
__version__ = '2.3.0'
__licence__ = 'Apache2.0'
__maintainer__ = 'Natarajan Chidambaram'
__email__ = 'natarajan.chidambaram@umons.ac.be'
__url__ = 'https://github.com/natarajan-chidambaram/RABBIT'
__description__ = 'RABBIT - Rabbit is an Activity Based Bot Identification Tool to detect bot contributors in GitHub repositories based on their recent activity sequences'
__long_description__ = 'This tool accepts the name of a contributor, a GitHub API key and its associated account name and computes its output in four steps.\\\
The first step consists of extracting the public events performed by contributors in GitHub repositories. This step results in a set of events.\\\
The second step identifies activities (belonging to 24 different activity types) performed by the contributor in GitHub repositories. If the number of activties is less than the maximum number of activities, then more events will be collected again. If the total number of activities (from all possible events) are less than specified minimum number of events, then the toll will exit with a message.\\\
The third step constitutes of identifying 38 contributor behavioural features, such as mean number of activities per activity type, number of activity types, median time between consecutive activities of different types, number of owners of repositories contributed to, Gini inequality of duration of consecutive activities and mean number of activities per repository.\\\
The forth step simply applies the BIMBAS model that we trained on 337K activities performed by 1035 bots and 1115 human contributors in GitHub repositories and gives the probability that a contributor is a bot.'
__classifiers__=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
__requirement__ = [
        'numpy==2.1.2',
        'pandas==2.2.3',
        'tqdm==4.66.5',
        'python-dateutil==2.9.0.post0',
        'scikit-learn == 1.5.2',
        'requests==2.32.3',
        'scipy==1.14.1',
        'urllib3==2.5.0',
        'joblib==1.4.2',
]

setup(
    name=__package__,

    version=__version__,

    description= __description__,
    long_description=__long_description__,

    url=__url__,

    maintainer=__maintainer__,
    maintainer_email=__email__,

    license=__licence__,

    classifiers=__classifiers__,

    keywords='github contributor activity',

    install_requires = __requirement__,

    # package_data={'':['rabbit_model.joblib']},
    include_package_data = True,
    packages = ['.'],

    entry_points={
        'console_scripts': [
            'rabbit=rabbit:cli',
        ]
    },

    py_modules=['rabbit'],
    zip_safe=True,
)
