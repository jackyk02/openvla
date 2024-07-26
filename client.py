import numpy as np
import requests
import json_numpy
from PIL import Image

json_numpy.patch()

image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = np.array(Image.open(requests.get(
    image_url, stream=True).raw).resize((256, 256)))

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": image,
          "instruction": "do something"}
).json()

print(action)
