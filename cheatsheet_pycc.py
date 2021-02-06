import sys
import numpy as np

read = sys.stdin.buffer.read
readline = sys.stdin.buffer.readline
readlines = sys.stdin.buffer.readlines

def main(H, MSE):
    N = len(MSE)
    # 終了時刻でソートしたバージョン
    ind = MSE[:, 2].argsort()
    MSE = MSE[ind]
    M, S, E = MSE[:, 0], MSE[:, 1], MSE[:, 2]
    # 種類ごとにソートしたバージョン
    sort_key = (M << 20) + E
    ind = np.argsort(sort_key)
    rank = np.argsort(ind)

    T = 100_010
    dp = np.zeros(T, np.int64)  # 終了時刻、種類はこだわらないので最大価値
    for t in range(1, T):
        dp[t] = max(dp[t], dp[t - 1])
        i1, i2 = np.searchsorted(E, [t, t + 1])
        for i in range(i1, i2):
            happy = dp[S[i]] + H[0]
            dp[t] = max(dp[t], happy)
            # 連続視聴による遷移を計算する
            k = 0
            while True:
                next_i = -1
                for j in range(rank[i] + 1, N):
                    j = ind[j]
                    if M[j] != M[i]:
                        break
                    if S[j] >= E[i]:
                        next_i = j
                        break
                if next_i == -1:
                    break
                k += 1
                i = next_i
                happy += H[k]
                dp[E[i]] = max(dp[E[i]], happy)
    return dp[-1]

if sys.argv[-1] == 'ONLINE_JUDGE':
    import numba
    from numba.pycc import CC
    i8 = numba.int64
    cc = CC('my_module')

    def cc_export(f, signature):
        cc.export(f.__name__, signature)(f)
        return numba.njit(f)

    main = cc_export(main, (i8[:], i8[:, :]))
    cc.compile()

from my_module import main

N = int(readline())
H = np.array(readline().split(), np.int64)
MSE = np.array(read().split(), np.int64).reshape(N, 3)

print(main(H, MSE))