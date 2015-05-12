import math

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
    vector_sub_checks_shapes(vector1, vector2)
    return [vector1[i] - vector2[i] for i in range(len(vector1))]


def vector_sum(*vectors):
    """vector_sum can take any number of vectors and add them together."""
    vector_sum_checks_shapes(vectors)
    sum = vectors[0]
    for vec in vectors[1:]:
        sum = vector_add(sum, vec)
    return sum


def dot(vector1, vector2):
    """
    dot([a b], [c d])   = a * c + b * d

    dot(Vector, Vector) = Scalar
    """
    dot_checks_shapes(vector1, vector2)
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


def vector_sub_checks_shapes(vector1, vector2):
    """Shape rule: the vectors must be the same size."""
    if shape_vectors(vector1) != shape_vectors(vector2):
        raise ShapeException


def vector_sum_checks_shapes(*vectors):
    """Shape rule: the vectors must be the same size."""
    test_shape = shape_vectors(vectors[0])
    for vec in vectors:
        if test_shape != shape_vectors(vec):
            raise ShapeException


def dot_checks_shapes(vector1, vector2):
    """Shape rule: the vectors must be the same size."""
    if shape_vectors(vector1) != shape_vectors(vector2):
        raise ShapeException


def magnitude(vector):
    """
    magnitude([a b])  = sqrt(a^2 + b^2)

    magnitude(Vector) = Scalar
    """
    return math.sqrt(sum([num * num for num in vector]))

def shape_matrices(matrix):
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    length = len(matrix)
    if isinstance(matrix[0], int):
        return (length,)
    else:
        return(length, (len(matrix[0])))

def matrix_row(matrix, row):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    return matrix[row]

def matrix_col(matrix, col):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    return[matrix[x][col] for x in range(len(matrix))]


def matrix_scalar_multiply(matrix, scalar):
    """
    [[a b]   *  Z   =   [[a*Z b*Z]
     [c d]]              [c*Z d*Z]]

    Matrix * Scalar = Matrix
    """
    return [vector_multiply(matrix[i], scalar) for i in range(len(matrix))]

def matrix_vector_multiply(matrix, vector):
    """
    [[a b]   *  [x   =   [a*x+b*y
     [c d]       y]       c*x+d*y
     [e f]                e*x+f*y]

    Matrix * Vector = Vector
    """
    matrix_vector_multiply_checks_shapes(matrix, vector)
    return [dot(matrix[i], vector) for i in range(len(matrix))]

def matrix_vector_multiply_checks_shapes(matrix, vector):
    """Shape Rule: The number of rows of the vector must equal the number of
    columns of the matrix."""
    if (shape_matrices(matrix))[0] != shape_vectors(vector):
        raise ShapeException


def matrix_matrix_multiply(matrix1, matrix2):
    """
    [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
     [c d]       [y z]]       [c*w+d*y c*x+d*z]
     [e f]                    [e*w+f*y e*x+f*z]]

    Matrix * Matrix = Matrix
    """
    matrix_matrix_multiply_checks_shapes(matrix1, matrix2)
    return [(dot((matrix_row(matrix1, i)), (matrix_col(matrix2, i))), \
            dot((matrix_row(matrix1, i)), (matrix_col(matrix2, (i+1))))) \
            for i in range(len(matrix1))]
    # return [matrix_vector_multiply(matrix2, matrix1[i]) for i in \
    #         range(len(matrix1))]

def matrix_matrix_multiply_checks_shapes(matrix1, matrix2):
    """Shape Rule: The number of columns of the first matrix must equal the
    number of rows of the second matrix."""
    if shape_matrices(matrix1)[0] != shape_matrices(matrix2)[1]:
        raise ShapeException
