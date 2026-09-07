"""
robot/franka.py
=================
Franka Panda forward-kinematics + dynamics accessors built on top of MuJoCo.
Contains single-sample FK (with Jacobians), baseline sequential FK batching,
and vectorized analytic batch FK for MPPI rollouts, along with hardware safety
guards and dynamic term accessors.
"""

from __future__ import annotations
import numpy as np
import mujoco

DOF = 7

# Hardware specifications from Franka Emika Panda datasheet
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=float)
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=float)
QD_MAX = np.array([2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100], dtype=float)  # rad/s
TAU_MAX = np.array([87.0,   87.0,    87.0,    87.0,    12.0,    12.0,    12.0], dtype=float)    # Nm

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]


def resolve_arm_indices(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """
    Resolves qpos, qvel, and actuator addresses by explicit joint/actuator names.
    Prevents slice bugs when gripper joints or auxiliary DOFs exist in MJCF.
    """
    qpos_adr = np.zeros(DOF, dtype=int)
    qvel_adr = np.zeros(DOF, dtype=int)
    for i, name in enumerate(ARM_JOINT_NAMES):
        try:
            j = model.joint(name)
        except KeyError as e:
            raise KeyError(
                f"Joint '{name}' missing from MJCF. Check ARM_JOINT_NAMES in franka.py."
            ) from e
        qpos_adr[i] = j.qposadr[0]
        qvel_adr[i] = j.dofadr[0]

    actuator_ids = np.zeros(DOF, dtype=int)
    for i, name in enumerate(ARM_ACTUATOR_NAMES):
        try:
            a = model.actuator(name)
        except KeyError as e:
            raise KeyError(
                f"Actuator '{name}' missing from MJCF. Check ARM_ACTUATOR_NAMES in franka.py."
            ) from e
        actuator_ids[i] = a.id

    return {"qpos_adr": qpos_adr, "qvel_adr": qvel_adr, "actuator_ids": actuator_ids}


class FrankaModel:
    """
    Unified kinematics and dynamics model for Franka Panda.
    Exposes single-sample FK, reference sequential batch FK, optimized 
    vectorized batch FK, and model dynamics (mass matrix, gravity, coriolis).
    """

    LINK_RADII_DEFAULT = [0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05]

    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, 
                 num_spheres_per_link: int = 2, link_radii: list[float] | None = None):
        self.model = mj_model
        self.data = mj_data
        self.n_per_link = num_spheres_per_link
        self.dof = DOF

        idx = resolve_arm_indices(mj_model)
        self.qpos_adr = idx["qpos_adr"]
        self.qvel_adr = idx["qvel_adr"]
        self.actuator_ids = idx["actuator_ids"]

        self.link_body_ids = [mj_model.body(f"link{i}").id for i in range(1, 8)]
        self.end_body_id = mj_model.body("hand").id
        
        radii = link_radii if link_radii is not None else self.LINK_RADII_DEFAULT
        self.sphere_radii = np.array(
            [radii[i] for i in range(len(self.link_body_ids)) for _ in range(self.n_per_link)],
            dtype=float
        )

        # Precompute static transforms and joint axes for vectorized FK
        self._init_vectorized_kinematics(mj_model)

    # --------------------------------------------------------------------------
    # Hardware Safety & Limit Enforcement
    # --------------------------------------------------------------------------
    def sanitize_q(self, q: np.ndarray) -> np.ndarray:
        """Clamps joint positions to legal physical hardware boundaries."""
        return np.clip(q, Q_MIN, Q_MAX)

    def sanitize_qdot(self, qdot: np.ndarray) -> np.ndarray:
        """Clamps joint velocities to physical hardware limits."""
        return np.clip(qdot, -QD_MAX, QD_MAX)

    def saturate_torques(self, tau: np.ndarray) -> np.ndarray:
        """Clamps commanded torques to hardware limits to prevent reflex shutdowns."""
        return np.clip(tau, -TAU_MAX, TAU_MAX)

    def is_within_limits(self, q: np.ndarray, margin: float = 0.02) -> bool:
        """Checks if configuration is within joint limits with safety margin (rad)."""
        return bool(np.all(q >= (Q_MIN + margin)) and np.all(q <= (Q_MAX - margin)))

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------
    def _set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr] = self.sanitize_q(q)

    def _set_qdot(self, qdot: np.ndarray) -> None:
        self.data.qvel[self.qvel_adr] = self.sanitize_qdot(qdot)

    def _segment_endpoints(self, bid_from: int, bid_to: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[bid_from].copy(), self.data.xpos[bid_to].copy()

    def num_spheres(self) -> int:
        return len(self.link_body_ids) * self.n_per_link

    # --------------------------------------------------------------------------
    # Vectorized Forward Kinematics Initialization
    # --------------------------------------------------------------------------
    def _init_vectorized_kinematics(self, mj_model: mujoco.MjModel) -> None:
        """Pre-extracts body offsets, parent IDs, and rotation axes from MuJoCo."""
        target_bids = self.link_body_ids + [self.end_body_id]
        bids_to_compute = set(target_bids)
        for bid in target_bids:
            curr = bid
            while curr != 0:
                bids_to_compute.add(curr)
                curr = mj_model.body_parentid[curr]

        self._fk_bids = sorted(list(bids_to_compute))
        self._fk_static_transforms = {}
        self._fk_joint_info = {}

        def quat2mat(q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            return np.array([
                [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
            ], dtype=float)

        for bid in self._fk_bids:
            T_static = np.eye(4)
            T_static[:3, :3] = quat2mat(mj_model.body_quat[bid])
            T_static[:3, 3] = mj_model.body_pos[bid]
            self._fk_static_transforms[bid] = T_static

            jnt_num = mj_model.body_jntnum[bid]
            jnt_adr = mj_model.body_jntadr[bid]

            arm_jnt_idx, jnt_pos, jnt_axis = None, None, None
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if mj_model.jnt_type[j] != 3:  # Hinge joint
                    continue
                qpos_adr_j = mj_model.jnt_qposadr[j]
                idx = np.where(self.qpos_adr == qpos_adr_j)[0]
                if len(idx) > 0:
                    arm_jnt_idx = idx[0]
                    jnt_pos = mj_model.jnt_pos[j].copy()
                    jnt_axis = mj_model.jnt_axis[j].copy()
                    break

            if arm_jnt_idx is not None:
                K = np.array([
                    [0, -jnt_axis[2], jnt_axis[1]],
                    [jnt_axis[2], 0, -jnt_axis[0]],
                    [-jnt_axis[1], jnt_axis[0], 0]
                ], dtype=float)
                self._fk_joint_info[bid] = {
                    'idx': arm_jnt_idx,
                    'pos': jnt_pos,
                    'K': K,
                    'K2': K @ K
                }

    # --------------------------------------------------------------------------
    # Forward Kinematics Implementations
    # --------------------------------------------------------------------------
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-configuration FK. Returns sphere centers (M, 3) and Jacobians (M, 3, 7).
        Used by GPMP2 and single-frame safety checks.
        """
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        next_ids = self.link_body_ids[1:] + [self.end_body_id]
        centers, jacs = [], []

        for bid, bid_next in zip(self.link_body_ids, next_ids):
            p_from, p_to = self._segment_endpoints(bid, bid_next)
            for k in range(self.n_per_link):
                alpha = (k + 0.5) / self.n_per_link
                p = (1 - alpha) * p_from + alpha * p_to
                centers.append(p)
                
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacp, jacr, p, bid)
                jacs.append(jacp[:, self.qvel_adr])

        return np.stack(centers), np.stack(jacs)

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """Sequential MuJoCo FK reference over batch Q (N, T, 7)."""
        N, T, _ = Q.shape
        M = self.num_spheres()
        out = np.zeros((N, T, M, 3))
        next_ids = self.link_body_ids[1:] + [self.end_body_id]

        for i in range(N):
            for t in range(T):
                self._set_q(Q[i, t])
                mujoco.mj_kinematics(self.model, self.data)
                idx = 0
                for bid, bid_next in zip(self.link_body_ids, next_ids):
                    p_from, p_to = self._segment_endpoints(bid, bid_next)
                    for k in range(self.n_per_link):
                        alpha = (k + 0.5) / self.n_per_link
                        out[i, t, idx] = (1 - alpha) * p_from + alpha * p_to
                        idx += 1
        return out

    def fk_batch_vectorized(self, Q: np.ndarray) -> np.ndarray:
        """
        Analytic vectorized FK over batch Q (N, T, 7).
        Computes homogeneous transforms in batch using pre-extracted kinematics parameters.
        """
        N, T, dof = Q.shape
        B = N * T
        Q_flat = Q.reshape(B, dof)

        world_transforms = {0: np.tile(np.eye(4), (B, 1, 1))}

        for bid in self._fk_bids:
            parent_id = self.model.body_parentid[bid]
            T_parent = world_transforms[parent_id]
            T_static = self._fk_static_transforms[bid]

            if bid in self._fk_joint_info:
                info = self._fk_joint_info[bid]
                q_j = Q_flat[:, info['idx']]

                sin_q = np.sin(q_j)[:, None, None]
                cos_q = np.cos(q_j)[:, None, None]

                R = np.eye(3) + sin_q * info['K'] + (1 - cos_q) * info['K2']
                p = info['pos']
                Rp = np.einsum('bij,j->bi', R, p)
                trans = p - Rp

                T_joint = np.zeros((B, 4, 4))
                T_joint[:, :3, :3] = R
                T_joint[:, :3, 3] = trans
                T_joint[:, 3, 3] = 1.0

                T_local = T_static @ T_joint
            else:
                T_local = T_static

            world_transforms[bid] = T_parent @ T_local

        endpoints = np.array([
            world_transforms[bid][:, :3, 3] 
            for bid in self.link_body_ids + [self.end_body_id]
        ])  # (8, B, 3)

        M = self.num_spheres()
        out = np.zeros((B, M, 3))

        idx = 0
        for i in range(len(self.link_body_ids)):
            p_from = endpoints[i]
            p_to = endpoints[i + 1]
            for k in range(self.n_per_link):
                alpha = (k + 0.5) / self.n_per_link
                out[:, idx] = (1 - alpha) * p_from + alpha * p_to
                idx += 1

        return out.reshape(N, T, M, 3)

    # --------------------------------------------------------------------------
    # Dynamic Term Accessors
    # --------------------------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Returns 7x7 inertia matrix M(q)."""
        self._set_q(q)
        mujoco.mj_forward(self.model, self.data)
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M_full)
        return M_full[np.ix_(self.qvel_adr, self.qvel_adr)]

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Returns 7x1 gravity torques g(q)."""
        self._set_q(q)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.qvel_adr].copy()

    def coriolis_times_qdot(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Returns 7x1 Coriolis and centrifugal forces C(q, qdot) * qdot."""
        self._set_q(q)
        self._set_qdot(qdot)
        mujoco.mj_forward(self.model, self.data)
        bias_arm = self.data.qfrc_bias[self.qvel_adr].copy()
        return bias_arm - self.gravity(q)
