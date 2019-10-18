class SAGGRiacSpec:

    def __init__(self, max_history, max_goals):
        self.max_history = max_history
        self.max_goals = max_goals


class GoalGANSpec:

    def __init__(self, state_noise_level, state_distance_threshold, buffer_size,
                 p_old_samples, p_context_samples, gan_pre_train_iters=1000, gan_train_iters=250, gan_train_interval=5):
        self.state_noise_level = state_noise_level
        self.state_distance_threshold = state_distance_threshold
        self.buffer_size = buffer_size
        self.p_old_samples = p_old_samples
        self.p_context_samples = p_context_samples
        self.gan_pre_train_iters = gan_pre_train_iters
        self.gan_train_iters = gan_train_iters
        self.gan_train_interval = gan_train_interval

        if p_context_samples > 0.5:
            raise RuntimeError("'p_context_samples' cannot be larger than 0.5")


class ExperimentSpec:

    def __init__(self, n_iter, init_dist, target_dist, value_features, init_policy, eta, alpha_schedule, sagg_riac_spec,
                 goal_gan_spec, n_samples=100, regularizer=1e-5, buffer_size=10, lower_variance_bound=None,
                 kl_div_thresh=None):
        self.n_iter = n_iter
        self.init_dist = init_dist
        self.target_dist = target_dist
        self.value_features = value_features
        self.init_policy = init_policy
        self.eta = eta
        self.alpha_schedule = alpha_schedule
        self.sagg_riac_spec = sagg_riac_spec
        self.goal_gan_spec = goal_gan_spec
        self.n_samples = n_samples
        self.regularizer = regularizer
        self.buffer_size = buffer_size
        self.lower_variance_bound = lower_variance_bound
        self.kl_div_thresh = kl_div_thresh

        if (self.kl_div_thresh is not None and self.lower_variance_bound is None) or \
                (self.kl_div_thresh is None and self.lower_variance_bound is not None):
            raise RuntimeError("kl_div_thresh and lower_variance_bound can only be specified together")
