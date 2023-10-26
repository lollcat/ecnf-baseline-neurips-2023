from functools import partial

from ecnf.cnf.sample_and_log_prob import sample_cnf, get_log_prob


sample_diff = partial(sample_cnf, t_final=1 - 1e-3)
get_log_prob_diff = partial(get_log_prob, t_final=1 - 1e-3)
