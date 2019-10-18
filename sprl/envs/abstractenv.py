import numpy as np
import copy
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, cost_fn_constructor, pid):
    parent_remote.close()
    cost_fn = cost_fn_constructor(pid)
    try:
        while True:
            mode, contexts, thetas = remote.recv()
            if mode == "experiment":
                n = contexts.shape[0]
                rewards = []
                successes = []
                for i in range(0, n):
                    r, s = cost_fn(contexts[i, :], thetas[i, :])
                    rewards.append(r)
                    successes.append(s)

                remote.send((rewards, successes))
            elif mode == "seed":
                cost_fn.set_seed(contexts)
                remote.send(True)
            else:
                raise RuntimeError("Unexpected mode '" + str(mode) + "'")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')


class AbstractEnvironment:

    def __init__(self, name, n_cores, default_setting, specs, cost_fn_constructor):
        self.name = name
        self.n_cores = n_cores
        self.setting = default_setting
        self.specs = specs

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_cores)])
        self.ps = [Process(target=worker, args=(work_remote, remote, cost_fn_constructor, pid))
                   for (work_remote, remote, pid) in zip(self.work_remotes, self.remotes, range(0, n_cores))]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def set_setting(self, name):
        if name not in self.specs.keys():
            raise RuntimeError("Invalid setting '" + str(name) + "'")
        self.setting = name

    def set_seed(self, seed):
        # We need to seed this process as well since the contexts and thetas get sampled in it
        np.random.seed(seed)
        # We need to seed the processes as well to get consistent samples
        count = 0
        for remote in self.remotes:
            remote.send(("seed", self.n_cores * seed + count, None))
            count += 1

        # This is just for synchronization
        [remote.recv() for remote in self.remotes]

    def get_spec(self):
        return copy.deepcopy(self.specs[self.setting])

    def get_name(self):
        return self.name + "-" + self.setting

    def evaluate(self, cs, xs):
        n = xs.shape[0]

        # We split the load of computing the experiment evenly among the cores
        contexts = []
        thetas = []
        n_sub = int(n / self.n_cores)
        rem = n - n_sub * self.n_cores
        count = 0
        for i in range(0, self.n_cores):
            extra = 0
            if rem > 0:
                extra = 1
                rem -= 1

            count_new = count + n_sub + extra

            sub_contexts = cs[count:count_new, :]
            if len(sub_contexts.shape) == 1:
                sub_contexts = np.array([sub_contexts])
            contexts.append(sub_contexts)

            sub_thetas = xs[count:count + n_sub + extra, :]
            if len(sub_thetas.shape) == 1:
                sub_thetas = np.array([sub_thetas])
            thetas.append(sub_thetas)

            count = count_new

        for remote, sub_contexts, sub_thetas in zip(self.remotes, contexts, thetas):
            remote.send(("experiment", sub_contexts, sub_thetas))
        tmp = [remote.recv() for remote in self.remotes]

        rewards = []
        successes = []
        for sub_rewards, sub_successes in tmp:
            rewards.append(sub_rewards)
            successes.append(sub_successes)

        return np.concatenate(rewards), np.concatenate(successes)

    def sample_rewards(self, n, policy, distribution=None):
        if distribution is None:
            distribution = self.specs[self.setting].target_dist

        # We split the load of computing the experiment evenly among the cores
        contexts = []
        thetas = []
        n_sub = int(n / self.n_cores)
        rem = n - n_sub * self.n_cores
        for i in range(0, self.n_cores):
            extra = 0
            if rem > 0:
                extra = 1
                rem -= 1
            sub_contexts = distribution.sample(n_sub + extra)
            if n_sub + extra == 1:
                sub_contexts = np.array([sub_contexts])
            contexts.append(sub_contexts)
            sub_thetas = []
            for j in range(0, n_sub + extra):
                sub_thetas.append(policy.sample_action(sub_contexts[j, :]))
            thetas.append(np.array(sub_thetas))

        for remote, sub_contexts, sub_thetas in zip(self.remotes, contexts, thetas):
            remote.send(("experiment", sub_contexts, sub_thetas))
        tmp = [remote.recv() for remote in self.remotes]

        rewards = []
        successes = []
        for sub_rewards, sub_successes in tmp:
            rewards.append(sub_rewards)
            successes.append(sub_successes)

        return np.concatenate(contexts), np.concatenate(thetas), np.concatenate(rewards), np.concatenate(successes)
