def compute_gp_weight(sigma_norm, eps=0.1, w_min=0.1, w_max=10.0):
    w = 1.0 / (sigma_norm**2 + eps)
    return max(w_min, min(w, w_max))


def use_gp_point(sigma_norm, sigma_high=2.0):
    return sigma_norm < sigma_high
