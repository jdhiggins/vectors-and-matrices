class ShapeException(Exception):
    pass

def shape_vectors(vector):
    length = len(vector)
    if isinstance(vector[0], int):
        return (length,)
    else:
        return((len(vector[0]), length))


def vector_add(vector1, vector2):
    """
    [a b]  + [c d]  = [a+c b+d]

    Matrix + Matrix = Matrix
    """
    vector_add_checks_shapes(vector1, vector2)
    return [vector1[i] + vector2[i] for i in range(len(vector1))]


def vector_sub(vector1, vector2):
    """
    [a b]  - [c d]  = [a-c b-d]

    Matrix + Matrix = Matrix
    """
    return [vector1[i] - vector2[i] for i in range(len(vector1))]


def vector_sum(*vectors):
    """vector_sum can take any number of vectors and add them together."""
    sum = vectors[0]
    for vec in vectors[1:]:
        sum = vector_add(sum, vec)
    return sum


def dot(vector1, vector2):
    """
    dot([a b], [c d])   = a * c + b * d

    dot(Vector, Vector) = Scalar
    """
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])


def vector_multiply(vector, scalar):
    """
    [a b]  *  Z     = [a*Z b*Z]

    Vector * Scalar = Vector
    """
    return [scalar * vector[i] for i in range(len(vector))]


def vector_mean(*vectors):
    """
    mean([a b], [c d]) = [mean(a, c) mean(b, d)]

    mean(Vector)       = Vector
    """
    return [vector_sum(*vectors)[i] / len(vectors) for i in range(len(vector_sum(*vectors)))]


def vector_add_checks_shapes(vector1, vector2):
    """Shape rule: the vectors must be the same size."""
    if shape_vectors(vector1) != shape_vectors(vector2):
        raise ShapeException
        

def magnitude(vector):
    """
    magnitude([a b])  = sqrt(a^2 + b^2)

    magnitude(Vector) = Scalar
    """
    # assert magnitude(m) == 5
    # assert magnitude(v) == math.sqrt(10)
    # assert magnitude(y) == math.sqrt(1400)
    # assert magnitude(z) == 0
