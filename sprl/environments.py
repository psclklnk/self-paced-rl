from sprl.envs.gate import Gate


def get(name, cores=1):
    if name == "gate":
        return Gate(name, cores)
    elif name == "reacher-obstacle":
        # We do this to avoid requiring mujoco-py if the reacher environments are not used
        from sprl.envs.reach_avoid import ReachAvoid
        return ReachAvoid(name, cores)
    elif name == "ball-in-a-cup":
        # We do this to avoid requiring mujoco-py if the ball in a cup environment is not used
        from sprl.envs.ball_in_a_cup import BallInACup
        return BallInACup(name, cores)
    elif name == "point-mass":
        # We do this to avoid requiring mujoco-py if the point maze environment is not used
        from sprl.envs.point_mass import PointMass
        return PointMass(name, cores)

    else:
        raise RuntimeError("Unknown environment '" + str(name) + "'")
