import tensorflow as tf
import tensorflow.keras.layers as kl

import cv2
from PIL import Image

import numpy as np

import argparse as arg
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# SRCNN
class SRCNN(tf.keras.Model):
    def __init__(self, h, w):
        super(SRCNN, self).__init__()

        self.conv1 = kl.Conv2D(64, 3, padding='same', activation='relu', input_shape=(None, h, w, 3))
        self.conv2 = kl.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = kl.Conv2D(3, 3, padding='same', activation='relu')

    def call(self, x):
        
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return h3


def main():
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='Super-resolution CNN prediction')
    parser.add_argument('--param', '-p', type=str, default=None,
                        help='学習済みパラメータの指定(未指定ならエラー)')
    parser.add_argument('--data_img', '-d', type=str, default=None,
                        help='画像ファイルの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result"),
                        help='保存先指定(デフォルト値=./result)')
    parser.add_argument('--he', '-he', type=int, default=256,
                        help='リサイズの高さ指定(デフォルト値=256)')
    parser.add_argument('--wi', '-wi', type=int, default=256,
                        help='リサイズの指定(デフォルト値=256)')
    parser.add_argument('--mag', '-m', type=int, default=2,
                        help='縮小倍率の指定(デフォルト値=2)')
    args = parser.parse_args()

    # パラメータファイル未指定時->例外
    if args.param is None:
        print("\nException: Trained Parameter-File not specified.\n")
        sys.exit()
    # 存在しないパラメータファイル指定時->例外
    if os.path.exists(args.param) is False:
        print("\nException: Trained Parameter-File {} is not found.\n".format(args.param))
        sys.exit()
    # 画像ファイル未指定時->例外
    if args.data_img is False:
        print("\nException: Image not specified.\n")
        sys.exit()
    # 存在しない画像ファイル指定時->例外
    if os.path.exists(args.data_img) is False:
        print("\nException: Image {} is not found.\n".format(args.data_img))
        sys.exit()
    # 幅高さ、縮小倍率いずれかに0が入力された時->例外
    if args.he == 0 or args.wi == 0 or args.mag == 0:
        print("\nException: Invalid value has been entered.\n")
        sys.exit()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Trained Prameter-File: {}".format(os.path.abspath(args.param)))
    print("# Image: {}".format(args.data_img))
    print("# Output folder: {}".format(args.out))
    print("")
    print("# Height: {}".format(args.he))
    print("# Width: {}".format(args.wi))
    print("# Magnification: {}".format(args.mag))
    print("===========================")

    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    # モデル構築
    model = SRCNN(args.he, args.wi)
    model.build((None, args.he, args.wi, 3))
    model.load_weights(args.param)

    # 入力画像加工(高解像画像)
    img = cv2.imread(args.data_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hr_img = cv2.resize(img, (args.he, args.wi))

    # 低解像度画像作成
    lr_img = cv2.resize(hr_img, (int(args.he / args.mag), int(args.wi / args.mag)))
    lr_img = cv2.resize(lr_img, (args.he, args.wi))
    lr_img_s = lr_img

    # 正規化
    lr_img = tf.convert_to_tensor(lr_img, np.float32)
    lr_img /= 255.0
    lr_img = lr_img[np.newaxis, :, :, :]

    # 超解像
    re = model.predict(lr_img)

    # データ加工
    re = np.reshape(re, (args.he, args.wi, 3))
    re *= 255
    re = np.clip(re, 0.0, 255.0)  # クリッピング(0~255に丸め込む)

    # 低解像度画像保存
    lr_img = Image.fromarray(np.uint8(lr_img_s))
    lr_img.show()
    lr_img.save(os.path.join(args.out, "Low-resolution Image(SRCNN).bmp"))

    # 超解像画像保存
    sr_img = Image.fromarray(np.uint8(re))
    sr_img.show()
    sr_img.save(os.path.join(args.out, "Super-resolution Image(SRCNN).bmp"))

    # 高解像度画像保存
    hr_img = Image.fromarray(np.uint8(hr_img))
    hr_img.show()
    hr_img.save(os.path.join(args.out, "High-resolution Image(SRCNN).bmp"))


if __name__ == "__main__":
    main()
