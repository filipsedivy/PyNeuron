def uniform_random(min_val, max_val, size):
    from random import uniform
    return [uniform(min_val, max_val) for _ in range(size)]