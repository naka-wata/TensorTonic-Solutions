import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet の最初の畳み込み層 Conv1 の出力形状をシミュレーションする関数。

    実際の畳み込み計算は行わず、
    出力と同じ shape を持つゼロ配列を返す。
    """

    # image.shape は (batch, height, width, channel) の形になっている
    # batch は「一度に処理する画像の枚数」
    batch = image.shape[0]

    # 入力画像の高さ
    image_h = image.shape[1]

    # 入力画像の幅
    image_w = image.shape[2]

    # 入力画像のチャンネル数
    # RGB画像なので通常は 3
    channel = image.shape[3]

    # Conv1 で使うカーネルサイズ
    # 11x11 の範囲を見ながら処理する
    kernel_size = 11

    # stride はカーネルを何マスずつ動かすか
    # stride=4 なので、4マスずつ移動する
    stride = 4

    # padding は画像の周りに追加する余白の大きさ
    # padding=2 なので、上下左右に2マス分の余白を足した
    padding = 2

    # filters はフィルターの数
    # 出力チャンネル数になる
    filters = 96

    # 畳み込み後の高さを計算する
    # // は整数割り算で、小数点以下を切り捨てる
    image_h_out = ((image_h + 2 * padding - kernel_size) // stride) + 1

    # 畳み込み後の幅を計算する
    # 高さと同じ式で計算できる
    image_w_out = ((image_w + 2 * padding - kernel_size) // stride) + 1

    # 実際の畳み込みは行わない
    # shape は (batch, 出力高さ, 出力幅, フィルター数)
    output = np.zeros((batch, image_h_out, image_w_out, filters))

    # 作成したゼロ配列を返す
    return output