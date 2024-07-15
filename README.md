# scut-eeg-feature-smooth

## Description

A package for  feature smoothing method based on EEG , contains 

1. move average filter
2. linear dynamic system
3. non-linear dynamic system Unscented Kalman Filter



## Installation

You can either git clone this whole repo by:

```
git clone https://github.com/didi226/scut_eeg_feature_smooth.git
cd scut_eeg_feature_smooth/dist
pip install scut_eeg_feature_smooth-0.0.1-py3-none-any.whl
```

## Usage

Simple demo for SSVEP detection methods.

```python
from scut_eeg_feature_smooth import feature_smooth
data = np.random.rand(60, 3, 500)
feature2 = feature_smooth(data,smooth_type="NDS-UKF",window_size=10)
```

## Cite 

The related article has not yet been published.