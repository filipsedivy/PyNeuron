def step_function(x):
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))


def softmax(vec):
    max_val = max(vec)
    exps = [pow(2.71828, v - max_val) for v in vec]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]
