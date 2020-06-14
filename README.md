# Super-resolution-CNN
Super-resolution CNN

詳細は[こちら(訓練フェーズ)](https://qiita.com/hima_zin331/items/7cdb6e12bcc85b683c26)と[こちら(推論フェーズ)](https://qiita.com/hima_zin331/items/ebb6046a2a8d860254e1)
*Not supported except in Japanese Language

___

## srcnn_tr.py

**Command**  
```
python srcnn_tr.py -d <DATA_DIR> -c <EPOCH_NUM> -b <BATCH_SIZE>
                                (-o <OUT_PATH> -he <HEIGHT> -wi <WIDTH> -m <MAG_SIZE>)
                                
EPOCH_NUM  : 3000 (Default)  
BATCH_SIZE : 32 (Default)  A, B, Cってなに？
OUT_PATH   : ./srcnn.h5 (Default)  
HEIGHT     : 256 (Default) *Input image size
WIDTH      : 256 (Default) *Input image size
MAG_SIZE   : 2 (Default) *See below for details.
```

**What are MAG_SIZE(-m)?**
The program creates low-resolution images from high-resolution images.  
Use `cv2.resize()` to resize the image to a smaller size and then restore it to its original size.  
By doing so, you can produce a low-resolution image.  
This is where MAG_SIZE comes into play. MAG_SIZE is the value that determines how much to shrink.

Like this!
```
        # Resize the width and height of the image divided by MAG_SIZE
        img_low = cv2.resize(img, (int(h/mag), int(w/mag)))
        # Resize to original size
        img_low = cv2.resize(img_low, (h, w))
```





