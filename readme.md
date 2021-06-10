## Installation
```shell
pip install .
```

## Run
```python
from pycurve.util import analyzeVideo
analyzeVideo(inDir='./data/210607_actuator.MOV', t0=2.0, dt=10.2)
```

``inDir``: the directory of the input video
``t0``: time of the first capture (second)
``dt``: time interval between two captures (second)

- A folder with the same name will be created at the current working directory. 
- Images of each capture are saved, whose name is the time of the capture
- ``data.txt`` records the number, time, and curvature radius(in pixel) of every capture
