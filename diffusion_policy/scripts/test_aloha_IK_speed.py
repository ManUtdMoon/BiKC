if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import h5py
import numpy as np
import modern_robotics as mr
import pathlib
import click
import time
import roboticstoolbox as rtb

from diffusion_policy.env.aloha.constants import (
    vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE
)

rev = 2 * np.pi
joint_ub = np.array([180., 101., 92., 180., 130., 180.]) / 180.0 * np.pi
joint_lb = np.array([-180., -101., -101., -180., -107., -180.]) / 180.0 * np.pi


def _qpos_regularization(qpos):
    qpos = qpos.copy()
    qpos = np.mod((qpos + np.pi), 2*np.pi) - np.pi
    
    return qpos


def _post_process_qpos(qpos, last_qpos):
    '''qpos[3] and qpos[5] is special because their ranges are [-pi, pi]
    so we want the them near the last time step to avoid a pi jump
    '''
    qpos = qpos.copy()
    if np.allclose(abs(last_qpos[3] - qpos[3]), np.pi, atol=1):
        assert np.allclose(abs(last_qpos[5] - qpos[5]), np.pi, atol=1), f"last: {last_qpos[5]:.4f} now: {qpos[5]:.4f}"
        qpos[3] = qpos[3] + np.pi if qpos[3] < 0 else qpos[3] - np.pi
        qpos[5] = qpos[5] + np.pi if qpos[5] < 0 else qpos[5] - np.pi
    return qpos


def ik_with_guesses(Slist, M, X, guesses):
    for guess in guesses:
        qpos, success = mr.IKinSpace(Slist, M, X, guess, 1e-4, 1e-4)
        solution_found = True

        if success:
            qpos = _qpos_regularization(qpos)
        else:
            solution_found = False
        
        if solution_found:
            return qpos, True
    
    return qpos, False


def rtb_ik(robot, X, guess):
    sol = robot.ikine_QP(
        X, end="/ee_gripper_link", q0=guess,
        tol=1e-4, mask=np.ones(6,), kq=1e1
    )
    return sol.q, sol.success


@click.command()
@click.option('--task', '-t',  required=True, default="sim_transfer_cube_scripted", type=str)
@click.option('--episode_id', '-i', default=0, type=int)
def main(task, episode_id):
    # init robot for calling its IK solver
    robot_l = rtb.models.URDF.vx300s()
    robot_r = rtb.models.URDF.vx300s()
    np.set_printoptions(
        suppress=True, precision=4, linewidth=150,
        formatter={'float': '{:>8.4f},'.format}
    )
    print(f"---------- task: {task:>10}, episode_id: {episode_id:>2} ----------")
    proj_dir = pathlib.Path(__file__).parent.parent.parent.expanduser()
    dataset_dir = proj_dir / "data" / "aloha" / "datasets" / task
    dataset_path = str(dataset_dir / f"episode_{episode_id}.hdf5")

    with h5py.File(dataset_path, "r") as f:
        qpos_gt = f["observations/qpos"][:]  # (T, 14)
    
    T = qpos_gt.shape[0]
    X_lb_lee = np.zeros((T, 4, 4), dtype=np.float32)
    X_rb_ree = np.zeros((T, 4, 4), dtype=np.float32)

    for t in range(T):
        X_lb_lee[t] = mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_gt[t, :6])
        X_rb_ree[t] = mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_gt[t, 7:13])
        # X_lb_lee[t] = robot_l.fkine(qpos_gt[t, :6], end="/ee_gripper_link")
        # X_rb_ree[t] = robot_r.fkine(qpos_gt[t, 7:13], end="/ee_gripper_link")

    qpos_IK = qpos_gt.copy()

    solving_time = list()
    q0 = np.zeros(6, dtype=np.float32)
    # q0 = np.array([0, -0.96, 1.16, 0, -0.3, 0], dtype=np.float32)
    guesses = [q0 for _ in range(3)]
    guesses[1][3] = +np.pi/3*2
    guesses[2][3] = -np.pi/3*2

    for t in range(1, T):
        t0 = time.time()
        idx = max(0, t-4)

        # # use mr
        guesses = [qpos_IK[idx,:6]]
        qpos_l, success_l = ik_with_guesses(vx300s.Slist, vx300s.M, X_lb_lee[t], guesses)
        guesses = [qpos_IK[idx,7:13]]
        qpos_r, success_r = ik_with_guesses(vx300s.Slist, vx300s.M, X_rb_ree[t], guesses)
        qpos_l = _post_process_qpos(qpos_l, qpos_IK[t-1,:6])
        qpos_r = _post_process_qpos(qpos_r, qpos_IK[t-1,7:13])

        # use rtb
        # qpos_l, success_l = rtb_ik(robot_l, X_lb_lee[t], np.random.rand(6,))
        # qpos_r, success_r = rtb_ik(robot_r, X_rb_ree[t], np.random.rand(6,))
        
        if (not success_l) or (not success_r):
            print(f"Failed to solve IK at t={t}")
            # exit the program
            return

        solving_time.append(time.time() - t0)
        
        qpos_IK[t, :6] = qpos_l
        qpos_IK[t, 7:13] = qpos_r
    
    solving_time = np.array(solving_time)
    print(f"solving time: {solving_time.mean():.5f}+-{solving_time.std():.5f}[s], max: {solving_time.max():.5f}")

    # average error
    qpos_gt = qpos_gt[1:] # (T-1, 14)
    qpos_IK = qpos_IK[1:] # (T-1, 14)
    error = np.abs(qpos_gt - qpos_IK).max(axis=1) # (T-1,)

    print(f"average error: {error.mean():.5f}+-{error.std():.5f}[rad], max: {error.max():.5f} @ t={np.argmax(error)+1}")
    max_error_idx = np.argmax(error)
    print(np.rad2deg(qpos_gt[max_error_idx]))
    print(np.rad2deg(qpos_IK[max_error_idx]))
    print(np.rad2deg(qpos_gt[max_error_idx-5]))
    print(np.rad2deg(qpos_IK[max_error_idx-5]))
    # print(np.rad2deg(qpos_gt[max_error_idx+1]))
    # print(np.rad2deg(qpos_IK[max_error_idx+1]))

    print(
        (   
            mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_gt[max_error_idx, :6])
            -mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_IK[max_error_idx, :6])
        )
    )
    
    print(
        (
            mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_gt[max_error_idx, 7:13])
            -mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_IK[max_error_idx, 7:13])
        )
    )

if __name__ == "__main__":
    main()