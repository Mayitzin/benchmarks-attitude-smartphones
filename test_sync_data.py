"""
On Attitude Estimation with Smartphones: Dataset
================================================

Load and organize the data acquired for the project "On Attitude Estimation
with Smartphones" mainly developed by Michel Thibaud as part of the Tyrex
research team of France's National Institute for Research in Computer Science
and Automation (Inria)

The Dataset provides attitude and position information of smartphones for its
analysis in attitude estimations, tracking and location. The sensorial data
acquisition was made with 3 common devices, whose technical details are:

+-----------+-----------------------------+-----------------------------+------------------+
| Model     | Accelerometer               | Gyroscope                   | Magnetometer     |
+===========+=============================+=============================+==================+
| iPhone 4S | STMicro STM33DH (100 Hz)    | STMicro AGDI (100 Hz)       | AKM 8975 (40 Hz) |
+-----------+-----------------------------+-----------------------------+------------------+
| iPhone 5  | STMicro LIS331DLH (100 Hz)  | STMicro L3G4200D (100 Hz)   | AKM 8963 (25 Hz) |
+-----------+-----------------------------+-----------------------------+------------------+
| Nexus 5   | InvenSense MPU6515 (200 Hz) | InvenSense MPU6515 (200 Hz) | AKM 8963 (60 Hz) |
+-----------+-----------------------------+-----------------------------+------------------+

Each sensor is an orthogonal tri-axial array embedded in the device. Although
these sensors are commercially available for anyone, each phone's attitude
estimation is not, and is referenced in the article as a "black box" estimation.

The iPhones provide estimated quaternions at the same time-stamp as their
gyroscopes. The Android device does not give synchronized data from its sensors,
but its accelerometers and gyroscopes have fairly similar sampling frequencies.

Additionally, the Android phone provides calibrated data of its gyroscope and
magnetometer. The iOS devices provide additional calibrated magnetometer data
only. In summary, the following data is obtained from each smartphone:

- `Accelerometer` in a N-by-4 array.
- `Gyroscope` in a N-by-4 array.
- `Calibrated Gyroscope` in a N-by-4 array. Android only.
- `Magnetometer` in a N-by-4 array.
- `Calibrated Magnetometer` in a N-by-4 array.
- `Attitude` as quaternions in a N-by-5 array.

The first column of each array is the time-stamp in seconds.

The phones are fitted into a custom handler with 5 infrared markers used by the
Qualisys to track its position and orientation at 150 Hz with a precision of
less than 0.5° of rotation. See https://kinovis.inria.fr/inria-platform/ for
more informtation about the setup. The Qualisys system provides a set of
different attitude data:

- `Positions` is a 3-by-N array of smartphone's positions in global coordinates.
- `Rotations` is a 9-by-N array with elements of the Direction Cosine Matrix
  specifying the orienation of the smartphone's frame.
- `RPYs` is a 3-by-N array with the roll-pitch-yaw angles determining the
  attitude matrix of the smartphone's frame.
- `Trajectories` is a dictionary with the 5 trajectories of the markers
  embedded to the handler. Each trajectory is a 4-by-N array, where the first
  column is the time-stamp.

Eight typical motions are recorded with each phone in a 10 x 10 m room at
Inria's facilities in Grenoble, France:

1. Augmented Reality.
2. Texting while walking.
3. Phoning while walking.
4. Swinging hand while walking.
5. In the front pocket while walking.
6. In the back pocket while walking.
7. In the front pocket while running.
8. In the hand while running.

See the main website (http://tyrex.inria.fr/mobile/benchmarks-attitude/) of the
dataset for further information.

References
----------
.. [1] Thibaud Michel, Pierre Genevès, Hassen Fourati, Nabil Layaïda. Attitude
    Estimation for Indoor Navigation and Augmented Reality with Smartphones.
    Pervasive and Mobile Computing, Elsevier, 2018, 46, pp.96-121.
    10.1016/j.pmcj.2018.03.004. hal-01650142v2
    (https://hal.inria.fr/hal-01650142v2/document)
.. [2] Thibaud Michel. On Attitude Estimation with Smartphones. 2016.
    http://tyrex.inria.fr/mobile/benchmarks-attitude/


.. moduleauthor:: Mario Garcia
"""

# Python Standard Library
import os
import datetime
from functools import reduce

# Third-party libraries
import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline

# AHRS
import ahrs
from ahrs.common.orientation import shepperd
from ahrs.common.orientation import q_prod
from ahrs.common.orientation import rotation
from ahrs.utils import WMM
from ahrs.utils import WGS

# from quickplots import plot
import ezview

DATASETS_PATH = 'datasets/'
FRAME = 'ENU'
# Inria's Laboratory Geodetic Coordinates
LATITUDE = 45.187778
LONGITUDE = 5.726945
HEIGHT = 0.2                            # in km
DATE = datetime.date(2016, 5, 31)       # Date of recordings
MAG_ELEMS = WMM(DATE, latitude=LATITUDE, longitude=LONGITUDE, height=HEIGHT).magnetic_elements
MAG_FIELD = np.array([MAG_ELEMS['X'], MAG_ELEMS['Y'], MAG_ELEMS['Z']])
MAGNETIC_DIP = MAG_ELEMS['I']
GRAVITY = WGS().normal_gravity(LATITUDE, HEIGHT*1000)

def DCM2q(dcm):
    if dcm.shape[-1]>3:
        if dcm.ndim<2:
            return shepperd(dcm.reshape((3, 3), order='F'))
        new_q = np.zeros((len(dcm), 4))
        for i, row in enumerate(dcm):
            if any(np.isnan(row)):
                new_q[i] = new_q[i-1].copy()
            else:
                qi = shepperd(row.reshape((3, 3), order='F'))
                new_q[i] = np.roll(qi, -1)
    else:
        if dcm.ndim>2:
            new_q = np.zeros((len(dcm), 4))
            for i, row in enumerate(dcm):
                new_q[i] = shepperd(row.reshape((3, 3), order='F'))
        else:
            return shepperd(dcm)
    new_q = remove_quaternion_jumps(new_q)
    return new_q

def remove_quaternion_jumps(quaternions):
    """
    Remove sudden flip of quaternion data

    Parameters
    ----------
    quaternions : NumPy array
        N-by-4 array with quaternion data.

    Returns
    -------
    new_quaternions : NumPy array
        N-by-4 array with corrected quaternion data.
    """
    if quaternions.ndim<2 or quaternions.shape[-1]!=4:
        raise ValueError("Input must be of shape (N, 4). Got {}".format(quaternions.shape))
    q_diff = np.diff(quaternions, axis=0)
    norms = np.linalg.norm(q_diff, axis=1)
    binaries = np.where(norms>1, 1, 0)
    nonzeros = np.nonzero(binaries)[0]
    jumps = nonzeros+1
    if len(jumps)%2:
        jumps = np.append(jumps, [len(q_diff)+1])
    jump_pairs = jumps.reshape((len(jumps)//2, 2))
    new_quaternions = quaternions.copy()
    for j in jump_pairs:
        new_quaternions[j[0]:j[1]] *= -1.0
    return new_quaternions

def slerp(q0: np.ndarray, q1: np.ndarray, t_array: np.ndarray, threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical Linear Interpolation between quaternions.

    Return a valid quaternion rotation at a specified distance along the minor
    arc of a great circle passing through any two existing quaternion endpoints
    lying on the unit radius hypersphere.

    Based on the method detailed in [Wiki_SLERP]_

    Parameters
    ----------
    q0 : NumPy array
        First endpoint quaternion.
    q1 : NumPy array
        Second endpoint quaternion.
    t_array : NumPy array
        Array of times to interpolate to.
    threshold : float, default: 0.9995
        Threshold to closeness of interpolation.

    Returns
    -------
    q : array
        New quaternion representing the interpolated rotation.

    References
    ----------
    .. [Wiki_SLERP] https://en.wikipedia.org/wiki/Slerp

    """
    qdot = np.dot(q0, q1)
    # Ensure SLERP takes the shortest path
    if qdot < 0.0:
        q1 *= -1.0
        qdot *= -1.0
    # Interpolate linearly (LERP)
    if qdot > threshold:
        result = q0[np.newaxis, :] + t_array[:, np.newaxis]*(q1 - q0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T
    # Angle between vectors
    theta_0 = np.arccos(qdot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - qdot*sin_theta/sin_theta_0
    s1 = sin_theta/sin_theta_0
    return s0[:, np.newaxis]*q0[np.newaxis, :] + s1[:, np.newaxis]*q1[np.newaxis, :]

def sync_data(ref_data, *data, **kw):
    """
    Synchronize data from their time-stamps

    Parameters
    ----------
    ref_data : array-like
        N-by-M array with reference time-stamps. If unidimensional, assumes
        first column contains the time-stamps.
    data : arrays
        N-by-M arrays with the data to synchronize. Assumes first columns has
        time-stamps.
    mode : str, default: 'linear'
        Mode of interpolation. Available modes are:

        'linear' (default)
            Interpolates the samples of the magnetometer to synchronize it to
            the sampling of accelerometers and gyroscopes.
        'cubic'
            Interpolates the samples of the magnetometer using a cubic spline.
        'downsample'
            Synchronizes samples to the times of the magnetometer, which is
            normally 4 times slower. This reduces the data other sensors by a
            factor of 4.
        'slerp'
            Synchronize quaternion data using SLERP.
    ratio : float, default: 0.85
        Ratio of intersection to discern.
    padding : str, default: 'fill'
        Padding type for array. Available options are:

        'fill' (default)
            Fills missing data with 'NaN' values.
        'clip'
            Crops arrays keeping the intersecting data only.

    Returns
    -------
    synch_data : list
        List of synchronized data arrays
    """
    ref_times = ref_data.copy() if ref_data.ndim == 1 else ref_data[:, 0]
    mode = kw.get('mode', 'linear')
    ratio = kw.get('ratio', 0.5)
    padding = kw.get('padding', 'fill')
    if len(data)<1:
        raise LookupError("You must provide data to synchronize")
    synch_data = [ref_data]
    for d in data:
        intersection, idx_1, idx_2 = np.intersect1d(ref_times, d[:, 0], assume_unique=True, return_indices=True)
        new_data = ref_data.copy()
        new_data[:, 1:] = np.nan
        if len(intersection)/len(ref_times)>ratio:
            new_data[idx_1] = d[idx_2]
        else:
            if mode.lower()=='linear':
                for i in range(1, 4):
                    new_data[:, i] = np.interp(new_data[:, 0], d[:, 0], d[:, i])
            elif mode.lower()=='cubic':
                for i in range(1, 4):
                    cs = CubicSpline(d[:, 0], d[:, i], extrapolate=False)
                    new_data[:, i] = cs(new_data[:, 0])
        synch_data.append(new_data)
    if padding.lower()=='clip':
        for i, d in enumerate(synch_data):
            synch_data[i] = d[~np.isnan(d).any(axis=1)]
        times = [d[:, 0] for d in synch_data]
        matched = reduce(np.intersect1d, times)
        indices = [np.where(np.in1d(t, matched))[0] for t in times]
        for i, d in enumerate(synch_data):
            synch_data[i] = d[indices[i]]
    return synch_data

class BenchmarkData:
    """
    Data structure corresponding to recordings of the session.

    MARG data is stored in independent TXT files of the corresponding
    session. Each file contains at least 4 columns representing the time
    stamp of each sample at the first columnd, followed by the values
    measured in the X-, Y- and Z-axis.

    MARG files appended with "-calibrated" add three columns, where the
    biases along each axis are repeated for each sample.

    The Quaternions given by the smartphones are also considered 'sensor
    measurements'

    Parameters
    ----------
    session : str
        Name of the session

    Attributes
    ----------
    session : str
        Name of the session
    smartphone : str {'ios', 'android'}
        Smartphone's OS used
    accelerometer : numpy.ndarray
        M-by-4 array with accelerometer data
    gyroscope : numpy.ndarray
        M-by-4 array with gyroscope data
    magnetometer : numpy.ndarray
        M-by-4 array with magnetometer data
    attitude : numpy.ndarray
        M-by-5 array with quaternion data from smartphone
    positions : numpy.ndarray
        M-by-3 array with three-dimensional position coordinates
    qsys_attitude : numpy.ndarray
        M-by-9 array with elements of 3-by-3 Direction Cosine Matrix
    trajectories : dict
        Dictionary with positions of the 5 tracked points of handler
    """
    def __init__(self, session):
        """

        """
        self.__dict__.update(dict().fromkeys(['accelerometer', 'gyroscope', 'magnetometer', 'attitude', 'positions', 'qsys_attitude', 'trajectories']))
        self.session = session
        self.load_sensors()
        self.get_frequencies()

    def load_sensors(self):
        if "calib" not in self.session:
            self.load_qualisys(DATASETS_PATH+f'qualisys/{self.session}.mat')
        self.smartphone = "ios" if "iphone" in self.session.lower() else "android"
        self.accelerometer = self.read_sensor("accelerometer")
        self.gyroscope = self.read_sensor("gyroscope", calibrated=True) if self.smartphone=='android' else self.read_sensor("gyroscope")
        self.magnetometer = self.read_sensor("magnetometer", calibrated=True)
        self.attitude = self.read_sensor("attitude")

    def read_sensor(self, sensor, calibrated: bool = False):
        """
        Read and store data from a sensor

        Parameters
        ----------
        sensor : str
            Sensor to get the information from. Options are: "accelerometer",
            "gyroscope", "magnetometer" and "attitude".
        calibrated : bool, default: False
            Whether to read the calibrated data of the given sensor.
        """
        # Input data verification
        if sensor.lower() not in ["accelerometer", "gyroscope", "magnetometer", "attitude"]:
            raise ValueError(f"Given sensor '{sensor}' is not included in dataset")
        if (sensor.lower() in ["accelerometer", "attitude"]) and calibrated:
            raise ValueError(f"There is no data of calibrated {sensor}s")
        if sensor.lower()=="gyroscope" and self.smartphone=="ios" and calibrated:
            raise ValueError("iPhone gyroscopes do not have calibrated data")
        # Path to data files
        path_to_folder = os.path.join(DATASETS_PATH, f"{self.smartphone}/{self.session}/")
        # Attitude Data from Smartphones
        if sensor == "attitude":
            if self.smartphone == "android":
                return np.genfromtxt(os.path.join(path_to_folder, "rotation-vector.txt"), usecols=[0, 4, 1, 2, 3])
            q = np.genfromtxt(os.path.join(path_to_folder, "attitude.txt"), usecols=[0, 4, 1, 2, 3])
            Rz90 = DCM2q(rotation('z', -90))    # -90 degree rotation about Z-axis
            for i, row in enumerate(q):
                q[i, 1:] = q_prod(Rz90, row[1:])
            return q
        # Each sensor Array
        is_calibrated = "-calibrated" if calibrated else ""
        fname = os.path.join(path_to_folder, f"{sensor.lower()}{is_calibrated}.txt")
        return np.genfromtxt(fname, usecols=[0, 1, 2, 3])

    def get_frequencies(self):
        """
        Estimate sampling frequencies from time-stamps

        Use the time-stamps of each property to estimate the difference between
        samples and invert it to get the sampling frequency.
        """
        self.freq_gyr = 1.0/np.diff(self.gyroscope[:, 0]).mean()
        self.freq_acc = 1.0/np.diff(self.accelerometer[:, 0]).mean()
        self.freq_mag = 1.0/np.diff(self.magnetometer[:, 0]).mean()
        self.freq_imu = (self.freq_acc + self.freq_gyr)/2.0
        self.freq_att = 1.0/np.diff(self.attitude[:, 0]).mean()

    def load_qualisys(self, path: str):
        """
        Read and store data from Qualisys recordings

        The Qualisys data is contained in a MAT file with the following
        structure:

            data_label :
                'File' : "path_to_file.qtm",
                'Timestamp' : "YYY-MM-DD, HH:MM:SS",
                'StartFrame' : 1 (scalar),
                'Frames' : N (scalar),
                'FrameRate' : 60 (scalar),
                'Trajectories' :
                    'Labeled' :
                        'Count' : M (scalar),
                        'Labels' : 1 x M matrix,
                        'Data' : M x 4 x N matrix
                    'Unidentified' :
                        'Count' : M (scalar),
                        'Labels' : 1 x M matrix,
                        'Data' : M x 4 x N matrix
                    'Discarded' :
                        'Count' : M (scalar),
                        'Labels' : 1 x M matrix,
                        'Data' : M x 4 x N matrix
                'RigidBodies' :
                    'Bodies' : B (scalar),
                    'Name' : "Name_of_Device",
                    'Positions' : 1 x 3 x N matrix,
                    'Rotations' : 1 x 9 x N matrix,
                    'RPYs' : 1 x 3 x N matrix,
                    'Residual' : 1 x 1 x N matrix

        This method updates the following attributes:

        * qsys_timestamp
        * label
        * positions
        * qsys_attitude
        * trajectories

        Parameters
        ----------
        path : str
            Path to MAT file.
        """
        mat_data = sio.loadmat(path)
        self.label = next(k for k in mat_data.keys() if not k.startswith('__'))
        N = mat_data[self.label]['Frames'][0, 0][0, 0]
        fps = 1.0/mat_data[self.label]['FrameRate'][0, 0][0, 0]
        self.qsys_timestamp = np.arange(0, N*fps, fps)
        # Trajectories of each marker
        self.trajectories = self.get_trajectories(mat_data[self.label]['Trajectories'][0, 0][0, 0])
        # Structure of Rigid Bodies
        self.positions = mat_data[self.label]['RigidBodies'][0, 0][0, 0]['Positions'][0].transpose()
        self.qsys_attitude = DCM2q(mat_data[self.label]['RigidBodies'][0, 0][0, 0]['Rotations'][0].transpose())
        self.badValue = np.where(np.isnan(self.qsys_attitude).sum(axis=1)>0, 1, 0)

    def get_trajectories(self, data, kind='Labeled'):
        labels = [x[0] for x in data[kind][0, 0]['Labels'][0]]
        arrays = [x.transpose() for x in data[kind][0, 0]['Data']]
        return dict(zip(labels, arrays))


if __name__ == "__main__":

    files = os.listdir(os.path.join(DATASETS_PATH, "qualisys"))
    sessions = [f.split('.')[0] for f in files if (f.endswith('.mat') and "iPhone" not in f)]
    # sessions = [f.split('.')[0] for f in files if f.endswith('.mat')]
    SESSION = sessions[1]
    print(f"Session: '{SESSION}'")
    data = BenchmarkData(SESSION)
    sdata = sync_data(data.gyroscope, data.accelerometer, data.magnetometer, padding='clip', mode='cubic')

    new_gyros = sdata[0][:, 1:]
    new_accs = sdata[1][:, 1:]
    new_mags = sdata[2][:, 1:]
    orientation = ahrs.filters.Tilt(new_accs, new_mags)
    orientation = ahrs.filters.Madgwick(new_gyros, new_accs, new_mags, frequency=data.freq_imu)
    # plot(new_accs, new_mags, data.qsys_attitude, tilt.Q)

    # ####### Attitude Estimation
    sdata_freq = 1.0/np.diff(sdata[0][:, 0]).mean()
    ref_attitude = np.c_[data.attitude[:, 0], remove_quaternion_jumps(data.attitude[:, 1:])]
    new_q = ahrs.QuaternionArray(orientation.Q)
    new_q.remove_jumps()

    # Plot
    ezview.plot_data(
        data.qsys_attitude,
        ref_attitude[:, 1:],
        new_q,
        subtitles=['Qualisys', 'Phone', 'EKF'],
        sharex=False)
