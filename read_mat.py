import os
import numpy as np
import ahrs
from benchmark import Qualisys
from benchmark import Phone
import ezview

def list_qualisys_sessions(dataset_path: str) -> list:
    """
    List sessions from dataset path.

    Parameters
    ----------
    dataset_path : str
        Path to dataset.

    Returns
    -------
    sessions : list
        List of sessions.
    """
    return [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mat')]

def find_session(substring: str, dataset_path: str) -> dict:
    """
    Find session from dataset path.

    Parameters
    ----------
    substring : str
        Substring to find.
    dataset_path : str
        Path to dataset.

    Returns
    -------
    found_sessions : dict
        Dictionary of found sessions, where the keys are the index and the
        values are the paths.

    Examples
    --------
    >>> found_sessions = find_session("Dist_AR", "/datasets/qualisys/")
    >>> for k, v in found_sessions.items():
    ...     print(k, v)
    0 ./datasets/qualisys/Guillaume_iPhone4S_Dist_AR.mat
    6 ./datasets/qualisys/Guillaume_iPhone4S_NoDist_AR.mat
    16 ./datasets/qualisys/Guillaume_Nexus5_NoDist_AR.mat
    24 ./datasets/qualisys/Jakob_iPhone4S_Dist_AR.mat
    """
    return {str(i): os.path.join(dataset_path, f) for i, f in enumerate(os.listdir(dataset_path)) if substring in f}

def list_invalid_recordings(qualisys_path: str) -> list:
    qualisys_sessions = list_qualisys_sessions(qualisys_path)
    for session in qualisys_sessions:
        try:
            qualisys = Qualisys(session)
            reference_rpy = np.unwrap(qualisys.RigidBodies['Rotations'].T, axis=0)
            valid_measurements = np.count_nonzero(np.all(~np.isnan(reference_rpy), axis=1))
            valid_ratio = valid_measurements / reference_rpy.shape[0]
            if valid_ratio > 99:
                print(f"{qualisys.session_name:<{50}}: {valid_measurements} [ COMPLETE ]")
            elif valid_ratio == 0:
                print(f"{qualisys.session_name:<{50}}: {valid_measurements} [ EMPTY ]")
            else:
                print(f"{qualisys.session_name:<{50}}: {valid_measurements} ({valid_ratio:.2%})")
        except ValueError:
            print(f"{os.path.basename(session):<{50}}: Invalid recording")

def main():
    root_path = "./datasets/"
    qualisys_files = list_qualisys_sessions(os.path.join(root_path, "qualisys/"))

    # Get one recording
    mat_file = qualisys_files[80]
    qualisys = Qualisys(mat_file)
    # qualisys.show_properties()
    print(qualisys.RigidBodies['DCM'].shape)
    reference_quaternions = ahrs.QuaternionArray(DCM=qualisys.RigidBodies['DCM'])
    print(reference_quaternions.shape)

    # scaling_factor = 0.001
    # trace3d = qualisys.RigidBodies['Positions'].T * scaling_factor
    # frames_idxs = np.linspace(0, qualisys.Frames-1, 10, dtype=int)
    # ezview.QPlot3D(
    #     trace3d,
    #     frames=zip(qualisys.RigidBodies['DCM'][frames_idxs], trace3d[frames_idxs]),
    #     )

    # reference_rpy = np.unwrap(qualisys.RigidBodies['RPYs'].T, axis=0)
    # invalid_measurements = np.count_nonzero(np.any(np.isnan(reference_rpy), axis=1))
    # invalid_ratio = invalid_measurements / reference_rpy.shape[0]
    # print(f"Invalid measurements: {invalid_measurements} ({invalid_ratio:.2%})")
    # reference_quaternions = ahrs.QuaternionArray(rpy=reference_rpy*ahrs.DEG2RAD)
    # # reference_quaternions.remove_jumps()

    # # Get Phone data
    # phone_path = os.path.join(root_path, "ios" if "iPhone" in qualisys.session_name else "android")
    # session_path = os.path.join(phone_path, qualisys.session_name)
    # phone = Phone(session_path)
    # phone.show_attributes()

    # ezview.plot_data(
    #     reference_quaternions,
    #     phone.acc[:, 1:], 
    #     phone.gyr[:, 1:],
    #     phone.mag[:, 1:],
    #     indices=[np.arange(reference_quaternions.shape[0]), phone.acc[:, 0], phone.gyr[:, 0], phone.mag[:, 0]],
    #     title="Sensors",
    #     ylabels=['', 'm/s^2', 'rad/s', 'uT'],
    #     sharex=False)

if __name__ == "__main__":
    main()
