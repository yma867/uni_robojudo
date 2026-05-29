import enum
import pickle

import numpy as np


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


def load_motion(file):
    with open(file, "rb") as filestream:
        in_dict = pickle.load(filestream)

        loop_mode_val = in_dict["loop_mode"]
        fps = in_dict["fps"]
        frames = in_dict["frames"]

        loop_mode = LoopMode(loop_mode_val)
        frames = np.array(frames, dtype=np.float32)

        motion_data = Motion(loop_mode=loop_mode, fps=fps, frames=frames)
    return motion_data


class Motion:
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
