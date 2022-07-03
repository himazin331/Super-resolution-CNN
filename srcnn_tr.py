from black import Any
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.python.keras import backend as K

import cv2
import numpy as np

import matplotlib.pyplot as plt

import argparse as arg
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TFメッセージ非表示


# SRCNN
class SRCNN(tf.keras.Model):
    def __init__(self, h: int, w: int):
        super(SRCNN, self).__init__()

        self.conv1 = L.Conv2D(
            64, 3, padding="same", activation="relu", input_shape=(None, h, w, 3)
        )
        self.conv2 = L.Conv2D(32, 3, padding="same", activation="relu")
        self.conv3 = L.Conv2D(3, 3, padding="same", activation="relu")

    def call(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return h3


# 学習
class trainer(object):
    def __init__(self, h: int, w: int):
        self.model = SRCNN(h, w)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[self._psnr],
        )

    def train(
        self,
        lr_imgs: tf.Tensor,
        hr_imgs: tf.Tensor,
        out_path: str,
        batch_size: int,
        epochs: int,
    ) -> tuple[Any, SRCNN]:  # ?[TypeHints] Any -> his: keras.callbacks.history
        # 学習
        # ?[TypeHints] his: keras.callbacks.history
        his = self.model.fit(lr_imgs, hr_imgs, batch_size=batch_size, epochs=epochs)

        print("___Training finished\n\n")

        # パラメータ保存
        print("___Saving parameter...")
        self.model.save_weights(out_path)
        print("___Successfully completed\n\n")

        return his, self.model

    # PSNR(ピーク信号対雑音比)
    def _psnr(self, h3, hr_imgs: tf.Tensor):
        return -10 * K.log(K.mean(K.flatten((h3 - hr_imgs)) ** 2)) / np.log(10)


# データセット作成
def create_dataset(
    data_dir: str, h: int, w: int, mag: int
) -> tuple[tf.Tensor, tf.Tensor]:
    print("\n___Creating a dataset...")

    prc: list[str] = ["/", "-", "\\", "|"]
    cnt: int = 0

    # 画像データの個数
    print("Number of image in a directory: {}".format(len(os.listdir(data_dir))))

    lr_imgs: tf.Tensor = []
    hr_imgs: tf.Tensor = []

    for c in os.listdir(data_dir):
        d: str = os.path.join(data_dir, c)

        ext: str
        _, ext = os.path.splitext(c)
        if ext.lower() == ".db":
            continue
        elif ext.lower() != ".bmp":
            continue

        # 読込、リサイズ(高解像画像)
        img: np.ndarray = cv2.imread(d)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))

        # 低解像度画像
        img_low: np.ndarray = cv2.resize(img, (int(w / mag), int(h / mag)))
        img_low = cv2.resize(img_low, (w, h))

        lr_imgs.append(img_low)
        hr_imgs.append(img)

        cnt += 1

        print(
            "\rLoading a LR-images and HR-images...{}    ({} / {})".format(
                prc[cnt % 4], cnt, len(os.listdir(data_dir))
            ),
            end="",
        )

    print(
        "\rLoading a LR-images and HR-images...Done    ({} / {})".format(
            cnt, len(os.listdir(data_dir))
        ),
        end="",
    )

    # 正規化
    lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32)
    lr_imgs = lr_imgs / 255
    hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
    hr_imgs = hr_imgs / 255

    print("\n___Successfully completed\n")
    return lr_imgs, hr_imgs


# PSNR, 損失値グラフ出力
def graph_output(history):  # ?[TypeHints] history: keras.callbacks.history
    # PSNRグラフ
    plt.plot(history.history["_psnr"])
    plt.title("Model PSNR")
    plt.ylabel("PSNR")
    plt.xlabel("Epoch")
    plt.legend(["Train"], loc="upper left")
    plt.show()

    # 損失値グラフ
    plt.plot(history.history["loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train"], loc="upper left")
    plt.show()


def main():
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description="Super-resolution CNN training")
    parser.add_argument(
        "--data_dir", "-d", type=str, default=None, help="画像フォルダパスの指定(未指定ならエラー)"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="パラメータの保存先指定(デフォルト値=./srcnn.h5",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="ミニバッチサイズの指定(デフォルト値=32)"
    )
    parser.add_argument(
        "--epoch", "-e", type=int, default=3000, help="学習回数の指定(デフォルト値=3000)"
    )
    parser.add_argument(
        "--he", "-he", type=int, default=256, help="リサイズの高さ指定(デフォルト値=256)"
    )
    parser.add_argument(
        "--wi", "-wi", type=int, default=256, help="リサイズの指定(デフォルト値=256)"
    )
    parser.add_argument("--mag", "-m", type=int, default=2, help="縮小倍率の指定(デフォルト値=2)")
    args = parser.parse_args()

    # 画像フォルダパス未指定->例外
    if args.data_dir is None:
        raise ValueError("Folder not specified.")
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.data_dir) is False:
        raise ValueError("Folder {} is not found.".format(args.data_dir))
    # 幅高さ、縮小倍率いずれかに0以下が入力された時->例外
    if args.he <= 0 or args.wi <= 0 or args.mag <= 0:
        raise ValueError("Invalid value has been entered.")

    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)
    out_path: str = os.path.join(args.out, "srcnn.h5")

    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(out_path))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("")
    print("# Height: {}".format(args.he))
    print("# Width: {}".format(args.wi))
    print("# Magnification: {}".format(args.mag))
    print("===========================\n")

    # データセット作成
    lr_imgs: tf.Tensor
    lr_imgs: tf.Tensor
    lr_imgs, hr_imgs = create_dataset(args.data_dir, args.he, args.wi, args.mag)

    # 学習開始
    print("___Start training...")
    Trainer = trainer(args.he, args.wi)

    # ?[TypeHints] his: keras.callbacks.history
    model: SRCNN
    his, model = Trainer.train(
        lr_imgs,
        hr_imgs,
        out_path=out_path,
        batch_size=args.batch_size,
        epochs=args.epoch,
    )

    # PSNR, 損失値グラフ出力、保存
    graph_output(his)


if __name__ == "__main__":
    main()
