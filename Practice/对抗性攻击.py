import sys
import math


def get_data():
    t = sys.stdin.readline().strip().split(' ')
    N, M = int(t[0]), int(t[1])

    X = [int(i) for i in sys.stdin.readline().strip().split(' ')]
    assert len(X) == N

    W1 = [float(t) for t in sys.stdin.readline().strip().split(' ')]
    assert len(W1) == M * N

    W2 = [float(t) for t in sys.stdin.readline().strip().split(' ')]
    assert len(W2) == 10 * M

    return X, W1, W2, N, M


def softmax(zlist):
    zmax = max(zlist)
    output = []
    sum = 0
    for z in zlist:
        sum = sum + (math.exp(z - zmax))
    for z in zlist:
        output.append((math.exp(z - zmax)) / sum)
    return output


def mm(A, B):  # HW x WC = HC
    H = len(A)
    W = len(B)
    assert len(A[0]) == W
    C = len(B[0])

    print(H, W, C)
    res_mat = [[i for i in range(C)] for _ in range(H)]

    for i in range(H):
        for j in range(C):
            for k in range(W):
                temp = A[i][k] * B[k][j]
                res_mat[i][j] += temp
    return res_mat


def print_m_shape(m):
    assert len(m[0])>0
    import numpy as np
    print(np.array(m[0]).flatten().shape)


def predict(X, W1, W2, M, N):
    W1_m = []  # M,N
    for i in range(1, M + 1):
        tmp = W1[(i - 1) * N + 1:i * N + 1]
        # print(tmp)
        W1_m.append(tmp)
    print_m_shape(W1_m)
    # print(len(W1_m[0]))

    X_m = []  # N,1
    for i in X:
        X_m.append([i])
    # print(len(X_m))
    print_m_shape(X_m)

    W1X = mm(W1_m, X_m)  # M,1
    print_m_shape(W1X)

    for i in range(M):
        if W1X[i][0] < 0:
            W1X[i][0] = 0

    W2_m = []  # 10,M
    for i in range(1, 10 + 1):
        tmp = W2[(i - 1) * M + 1:i * M + 1]
        W2_m.append(tmp)

    W2X = mm(W2_m, W1X)  # 10, 1
    W2X = [i[0] for i in W2X]

    max_num = max(W2X)
    max_id = W2X.index(max_num)
    # print(max_id, max_num)
    return max_id, max_num


def predict_np(X, W1, W2, M, N):
    import numpy as np
    X = np.array(X).reshape(N, 1)
    W1 = np.array(W1).reshape(M, N)
    W2 = np.array(W2).reshape(10, M)
    relu = np.maximum(np.dot(W1, X), 0)
    output = softmax(np.dot(W2, relu).flatten().tolist())
    # print(output)
    max_num = max(output)
    max_id = output.index(max_num)
    # print(max_id, max_num)
    return max_id, max_num


def gen_number():
    ran = [i for i in range(1, 128)]
    ran = ran + [-1 * i for i in range(1, 128)]
    ran.append(0)
    ran.append(-128)
    # print(ran)
    return ran


def get_data2():
    MN = '8 4'
    X = '-4 -71 -56 -41 85 -19 -56 -3'
    W1 = '0.00719 0.01590 -0.01121 -0.02345 0.00777 0.01680 0.01642 -0.01437 0.04963 -0.02698 -0.03168 -0.02930 0.00784 -0.03372 -0.01824 0.01997 -0.01687 -0.02018 -0.00434 -0.00647 -0.01860 -0.01780 -0.01345 0.03369 0.00142 -0.00109 -0.02072 0.00518 -0.02600 -0.01217 -0.00510 -0.00254'
    W2 = '-0.00372 0.06219 0.00260 0.06550 -0.02418 -0.02375 0.00115 0.00132 0.00280 -0.01428 0.02612 -0.03527 -0.02926 -0.02194 -0.04160 0.03126 0.01071 0.02239 0.00883 0.03610 0.00117 0.00429 -0.05671 0.00374 0.03496 0.03749 0.03426 0.01259 0.01202 -0.00021 -0.04738 -0.02131 0.02525 0.04419 -0.01626 0.04310 -0.01328 -0.00932 -0.03152 0.06103'

    t = MN.strip().split(' ')
    N, M = int(t[0]), int(t[1])

    X = [int(i) for i in X.strip().split(' ')]
    assert len(X) == N

    W1 = [float(t) for t in W1.strip().split(' ')]
    assert len(W1) == M * N

    W2 = [float(t) for t in W2.strip().split(' ')]
    assert len(W2) == 10 * M

    return X, W1, W2, N, M


def find_sensitive():
    # X, W1, W2, N, M = get_data()

    X, W1, W2, N, M = get_data2()

    init_id, init_num = predict(X, W1, W2, M, N)
    # init_id, init_num = predict_np(X, W1, W2, M, N)
    # print(init_id, init_num)

    same_min_num, same_min_id = 10, 0
    diff_max_num, diff_max_id = 0, 0
    out1, out2 = 0, 0

    ran = gen_number()

    for i in range(N):
        for j in ran:
            X_tmp = X[:]
            X_tmp[i] = j
            new_id, new_num = predict_np(X_tmp, W1, W2, M, N)
            new_id, new_num = predict(X_tmp, W1, W2, M, N)
            # print(new_id, new_num)

            # if new_id == init_id:
            #     if new_num < same_min_num:
            #         same_min_num = new_num
            #         same_min_id = id
            #         out1 = i + 1
            #         out2 = j
            if new_id != init_id:
                if new_num > diff_max_num:
                    diff_max_num = new_num
                    diff_max_id = id
                    out1 = i + 1
                    out2 = j

    print(out1, out2)


if __name__ == '__main__':
    find_sensitive()
