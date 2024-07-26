import numpy as np
import requests
import json_numpy
json_numpy.patch()

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8),
          "instruction": "do something"}
).json()
