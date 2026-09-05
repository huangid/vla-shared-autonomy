import sys
import threading
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import hid



def _to_int16(lo, hi):
    v = lo | (hi << 8)
    return v - 65536 if v >= 32768 else v



class SpaceMousePilot:
    def __init__(self, cfg_path=None, data_path=None, num_envs=1,
                 device="cpu", replay_mode=False,
                 pos_scale=0.15, rot_scale=0.3, vendor=0x256f, product=0xc635):
        self._device = device
        self._num_envs = num_envs
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self._pos = np.zeros(3)
        self._rot = np.zeros(3)
        self._grip = -1.0
        self._lock = threading.Lock()
        self._dev = hid.device()
        self._dev.open(vendor, product)
        self._dev.set_nonblocking(True)
        print("SpaceMousePilot opened:", self._dev.get_product_string(),
              file=sys.stderr, flush=True)
        self._run = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._run:
            data = self._dev.read(64)
            if data:
                rid = data[0]
                if rid == 1 and len(data) >= 7:
                    x = -_to_int16(data[1], data[2]) / 350.0
                    y = _to_int16(data[3], data[4]) / 350.0
                    z = -_to_int16(data[5], data[6]) / 350.0
                    with self._lock:
                        self._pos = np.array([x, y, z])
                elif rid == 2 and len(data) >= 7:
                    rx = -_to_int16(data[1], data[2]) / 350.0
                    ry = _to_int16(data[3], data[4]) / 350.0
                    rz = -_to_int16(data[5], data[6]) / 350.0
                    with self._lock:
                        self._rot = np.array([rx, ry, rz])
                elif rid == 3:
                    with self._lock:
                        if data[1] & 0x01:      # left button -> close (big grip closed on object)
                            self._grip = 1.0
                        elif data[1] & 0x02:    # right button -> open (release)
                            self._grip = -1.0
            else:
                time.sleep(0.002)

    def reset(self):
        with self._lock:
            self._pos = np.zeros(3)
            self._rot = np.zeros(3)

    def clear(self, env_ids=None):
        self.reset()

    def get_actions(self, eidx, pos, quat=None, grip=None, verbose=False):
        N = pos.shape[0]
        with self._lock:
            dpos = torch.tensor(self._pos * self.pos_scale,
                                dtype=torch.float32, device=self._device)
            drot = self._rot * self.rot_scale
            gval = self._grip

        print("SM cmd:", dpos.cpu().numpy(), drot, file=sys.stderr, flush=True)

        dz = 0.005
        dpos[dpos.abs() < dz] = 0.0
        drot_arr = np.where(np.abs(drot) < dz, 0.0, drot)
        self.is_idle = bool((dpos.abs() < dz).all()) and bool((np.abs(drot_arr) < dz).all())

        new_pos = pos + dpos.unsqueeze(0).expand(N, -1)

        if quat is None:
            quat = torch.zeros((N, 4), device=self._device)
            quat[:, 0] = 1.0

        dq = Rotation.from_rotvec(drot_arr).as_quat()
        dq = torch.tensor([dq[3], dq[0], dq[1], dq[2]],
                          dtype=torch.float32, device=self._device)
        new_quat = self._quat_mul(dq.unsqueeze(0).expand(N, -1), quat)
        new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        new_grip = torch.full((N, 1), gval, dtype=torch.float32, device=self._device)
        return torch.cat([new_pos, new_quat, new_grip], dim=-1)

    @staticmethod
    def _quat_mul(a, b):
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dim=-1)
