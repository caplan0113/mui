# generate_data.py
import json
from lib import *
from scipy.spatial.transform import Rotation as R

allData = load_all()
data = allData[3][0][0]
pos = data["pos"]
rot = data["rot"]
time = data["time"]

base_vec = [0, 0, 1]
vec = [R.from_euler("xyz", r, degrees=True).apply(base_vec) for r in rot]

frames = [{"x": p[0], "z": p[2], "vx": v[0], "vz": v[2], "time": t} for p, v, t in zip(pos, vec, time)]

with open("project/vector_data.json", "w") as f:
    json.dump(frames, f)
