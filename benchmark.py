"""
Attitude Benchamark with Smartphones
"""

import os
import scipy.io as spio
import numpy as np

class Qualisys:
    """
    Class for reading Qualisys mat files

    Parameters
    ----------
    path : str
        Path to mat file.

    Attributes
    ----------
    mat : dict
        Dictionary of mat file.
    header : str
        Header of mat file.
    version : str
        Version of mat file.
    globals : str
        Globals of mat file.
    session_name : str
        Name of session.
    session : dict
        Dictionary of session.
    """
    def __init__(self, path: str) -> None:
        self.mat : str = spio.loadmat(path)
        self.header : str = self.mat['__header__']
        self.version : str = self.mat['__version__']
        self.globals : str = self.mat['__globals__']
        # Define recording properties
        self.session_name : str = [x for x in self.mat.keys() if '__' not in x][0]
        split_values = self.session_name.split('_')
        if len(split_values) != 4:
            raise ValueError(f"Invalid session name: {self.session_name}")
        self.user, self.model, self.dist, self.motion = split_values
        # Get field names
        self.session = self.load_session(self.mat[self.session_name][0][0])
        self.File = self.session["File"]
        self.Timestamp = self.session["Timestamp"]
        self.StartFrame = self.session["StartFrame"]
        self.Frames = self.session["Frames"]
        self.FrameRate = self.session["FrameRate"]
        # Get Data
        self.Trajectories = self.load_trajectories(self.session["Trajectories"])
        self.RigidBodies = self.load_rigid_bodies(self.session["RigidBodies"])

    def show_properties(self):
        for item in dir(self):
            if not item.startswith('__'):
                attr = self.__getattribute__(item)
                if isinstance(attr, np.ndarray):
                    attr = attr.shape
                elif isinstance(attr, dict):
                    attr = list(attr.keys())
                elif callable(attr):
                    continue
                print('{:<{w}}  {}'.format(item, attr, w=len(max(dir(self), key=len))))

    def load_session(self, array):
        keys = ["File", "Timestamp", "StartFrame", "Frames", "FrameRate", "Trajectories", "RigidBodies"]
        d = {}
        for i, k in enumerate(keys):
            val = array[i][0]
            if isinstance(val, str):
                val = val.strip()
            if len(val) == 1:
                val = val[0]
            d.update({k: val})
        return d

    def load_trajectories(self, array):
        traj_keys = ["Labeled", "Unidentified", "Discarded"]
        d_traj = {}
        for i, k in enumerate(array):
            val = k[0][0]
            if np.squeeze(val[0])<1:
                d_traj.update({traj_keys[i]: None})
            else:
                count = val["Count"][0][0]
                labels = val["Labels"][0]
                data = val["Data"]
                d_traj.update({traj_keys[i]: {"count": count, "labels": labels, "data": data}})
        return d_traj

    def load_rigid_bodies(self, array):
        rb_keys = ["Bodies", "Name", "Positions", "Rotations", "RPYs", "Residual"]
        d_rb = {}
        for i, v in enumerate(array):
            val = v[0]
            if val.ndim < 2:
                if isinstance(val[0], (int, float)):
                    d_rb.update({rb_keys[i]: val[0]})
                if hasattr(val[0], '__iter__'):
                    d_rb.update({rb_keys[i]: val[0][0]})
            else:
                d_rb.update({rb_keys[i]: val})
        # Store rotations as 3-by-3 Direction Cosine Matrices
        d_rb['DCM'] = d_rb['Rotations'].T.reshape((-1, 3, 3))
        return d_rb

class Phone:
    def __init__(self, path):
        self.path = path
        files = os.listdir(self.path)
        prop_files = [os.path.join(self.path, x) for x in files if x.endswith(".properties")]
        self.pfile = prop_files[0] if len(prop_files) > 0 else None
        self.os = "ios" if "iphone" in os.path.basename(path).lower() else "android"
        self.get_device_properties()
        self.acc = np.genfromtxt(os.path.join(path, "accelerometer.txt"), dtype=float)
        self.gyr = np.genfromtxt(os.path.join(path, "gyroscope.txt"), dtype=float)
        self.mag = np.genfromtxt(os.path.join(path, "magnetometer.txt"), dtype=float)

    def get_device_properties(self):
        self.model = 'Nexus 5' if self.os == 'android' else ''
        for s in os.path.basename(self.path).split('_'):
            if "iphone" in s.lower():
                self.model = s
                break
        if self.pfile:
            with open(self.pfile, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "=" in line:
                    split_line = line.strip().split("=")
                    self.__dict__.update({split_line[0].strip(): split_line[1].strip()})

    def show_attributes(self):
        phone_properties = [x for x in dir(self) if not (hasattr(self.__getattribute__(x), '__call__') or x.startswith('__'))]
        for p in phone_properties:
            p_attr = self.__getattribute__(p)
            if isinstance(p_attr, np.ndarray):
                p_attr = p_attr.shape
            print('{:<{w}}  {}'.format(p, p_attr, w=len(max(phone_properties, key=len))))

if __name__ == '__main__':
    raise RuntimeError(f"'{__file__}' should not be run directly!")