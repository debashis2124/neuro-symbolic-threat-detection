def compute_hybrid_score(f_nn, g_sr, alpha=0.6, beta=0.4):
    return alpha * f_nn + beta * g_sr