import setuptools
from setuptools import find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="scut_eeg_feature_smooth",
    version="0.0.1",
    author="Di Chen",
    author_email="3517725675@qq.com",
    description="EEG feature smooth",
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={"": "src"},
    install_requires=[
        'requests',
        'importlib-metadata; python_version >= "3.8"',
        'einops',
        'mne_features',
        'EMD-signal',
        'scipy',
        'PyWavelets',
        'pyts',
        'antropy',
        'pyentrp',
        'tftb',
        'statsmodels',
        'scipy',
        'emd-signal',
        'pyentrp',
        'nolds',
        'pactools',
        'EntropyHub',
'pykalman',
'mne_connectivity',
'spkit',
'fooof',
'nilearn'
],
)
