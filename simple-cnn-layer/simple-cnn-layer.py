import numpy as np

def conv2d(x, W, b):
    batch = x.shape[0]       # バッチサイズ
    C_out = W.shape[0]       # フィルタの個数 = 出力チャネル数
    H_out = x.shape[2] - W.shape[2] + 1
    W_out = x.shape[3] - W.shape[3] + 1
    output = np.zeros((batch, C_out, H_out, W_out))

    for n in range(batch):           # バッチごとにループ
        for f in range(C_out):       # フィルタごとにループ
            for h in range(H_out):
                for w in range(W_out):
                    total = 0
                    for c in range(x.shape[1]):  # 入力チャネルごとにループ
                        patch = x[n, c, h:h+W.shape[2], w:w+W.shape[3]]
                        total += np.sum(patch * W[f, c, :, :])
                    output[n, f, h, w] = total + b[f]

    return output