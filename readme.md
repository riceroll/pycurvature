# pyCurvature
Use opencv to detect curve and analyze curvatures from video/image.

## Installation
```shell
pip install .
```

## Example
```python
from pycurve.util import analyzeVideo, getDefaultSetting
analyzeVideo(inDir='./data/0615.MOV', t0=2.0, dt=10.2)
```

## Run
```python
from pycurve.util import analyzeVideo, getDefaultSetting
setting = getDefaultSetting()   # get the defaultSetting dict, check function definition for detail
print(setting)                  # check available settings
setting['hue'] = 65             # overwrite default Setting
setting['alpha'] = 2.6
analyzeVideo(inDir='./data/0615.MOV', 
             t0=2.0, 
             dt=10.2,
             setting=setting,
             outDir='.',
             testing=False
             )
```

``inDir``: the directory of the input video

``t0``: time of the first capture (second)

``dt``: time interval between two captures (second)

``outDir``: directory for saving data

``testing``: true if in testing mode, where processing figures will be shown

- A folder with the same name will be created at the current working directory. 
- Images of each capture are saved, whose name is the time of the capture
- ``data.txt`` records the number, time, and curvature radius(in pixel) of every capture
