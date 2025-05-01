def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))


def vector_add(vec1, vec2):
    return [x + y for x, y in zip(vec1, vec2)]


def scalar_multiply(scalar, vec):
    return [scalar * x for x in vec]


