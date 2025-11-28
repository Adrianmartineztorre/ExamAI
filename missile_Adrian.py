import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    R: np.ndarray
    z: np.ndarray

def kalman_filter(s: KalmanState) -> Tuple[KalmanState, np.ndarray]:
    s.x = s.F @ s.x
    s.P = s.F @ s.P @ s.F.T
    K = s.P @ np.linalg.inv(s.P + s.R)
    s.x = s.x + K @ (s.z - s.x)
    s.P = s.P - K @ s.P
    return s, K

def missile(o, sigma):
    F = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    R = np.diag([
        sigma**2, sigma**2, sigma**2,
        2*sigma**2, 2*sigma**2, 2*sigma**2
    ])

    s = KalmanState(
        x=np.array([o[0,0], o[1,0], o[2,0], 0, 0, 0]),
        z=np.zeros(6),
        F=F,
        R=R,
        P=R.copy()
    )

    for t in range(1, o.shape[1]):
        vx = o[0,t] - o[0,t-1]
        vy = o[1,t] - o[1,t-1]
        vz = o[2,t] - o[2,t-1]
        s.z = np.array([o[0,t], o[1,t], o[2,t], vx, vy, vz])
        s, K = kalman_filter(s)

    return s.x

def main():
    o = np.loadtxt(
        r"C:\Users\quinc\Downloads\missile_data.csv",
        delimiter=",",
        skiprows=1
    ).T

    sigma = 0.5
    state = missile(o, sigma)

    print(f"X:  {state[0]:.2f}")
    print(f"Y:  {state[1]:.2f}")
    print(f"Z:  {state[2]:.2f}")
    print(f"Vx: {state[3]:.3f}")
    print(f"Vy: {state[4]:.3f}")
    print(f"Vz: {state[5]:.3f}")

if __name__ == "__main__":
    main()
