from numba import jit


@jit
def array_qr(arr):
    """Calculate R of a QR decomposition for matrices in an array.

    args:
        arr (np.ndarray): 3d array of [nmixtures * nind, m, n], where m >= n.
            It is overwritten wtih the  R of the QR decomposition.

    The algorithm uses Givens Rotations for the triangularization and fully
    exploits the sparseness of the Rotation Matrices.

    Due to the orthogonality of the Q Matrix, the (input) m x n matrices A and
    the output n x n matrices R are related as follows:

    R'R = A'A

    According to some definitions, this makes R the transpose of the cholesky
    factor of A'A. Other definitions of cholesky would require that all
    diagonal elements are positive. This requirement also makes the QR
    decomposition unique. It is achieved by setting make_unique to True.

    For the Unscented Kalman filter and the lower triangular update algorithm
    the weaker definition of cholesky is usually sufficient.

    The function is based on the following algorithm found at stackoverflow,
    but tested for correctness and optimized for speed::

        function [Q,R] = qrgivens(A)
            [m,n] = size(A);
            Q = eye(m);
            R = A;

            for j = 1:n
                for i = m:-1:(j+1)
                    G = eye(m);
                    [c,s] = givensrotation( R(i-1,j),R(i,j) );
                    G([i-1, i],[i-1, i]) = [c -s; s c];
                    R = G'*R;
                    Q = Q*G;
              end
            end
        end

        function [c,s] = givensrotation(a,b)
            if b == 0
                c = 1;
                s = 0;
            else
                if abs(b) > abs(a)
                    r = a / b;
                    s = 1 / sqrt(1 + r^2);
                    c = s*r;
            else
                r = b / a;
                c = 1 / sqrt(1 + r^2);
                s = c*r;
            end
          end

        end

    """
    long_side, m, n = arr.shape
    for u in range(long_side):
        for j in range(n):
            for i in range(m - 1, j, -1):
                b = arr[u, i, j]
                if b != 0.0:
                    a = arr[u, i - 1, j]
                    if abs(b) > abs(a):
                        r = a / b
                        s = 1 / (1 + r ** 2) ** 0.5
                        c = s * r
                    else:
                        r = b / a
                        c = 1 / (1 + r ** 2) ** 0.5
                        s = c * r
                    for k in range(n):
                        helper1 = arr[u, i - 1, k]
                        helper2 = arr[u, i, k]
                        arr[u, i - 1, k] = c * helper1 + s * helper2
                        arr[u, i, k] = -s * helper1 + c * helper2

    return arr
