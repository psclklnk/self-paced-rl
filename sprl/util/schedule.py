import numpy as np


class PercentageSchedule:

    def __init__(self, percentage, offset=0, max_alpha=1e10):
        self._percentage = percentage
        self._offset = offset
        self._max_alpha = max_alpha

    def get_value(self, t, avg_rew, kl_div):
        if t > self._offset:
            if kl_div <= 0:
                return self._max_alpha
            else:
                alpha = (self._percentage * np.abs(avg_rew)) / kl_div
                if alpha >= self._max_alpha:
                    alpha = self._max_alpha

                return alpha
                # return np.minimum((self._percentage * np.abs(avg_rew)) / kl_div, self._max_alpha)
        else:
            return 0.
