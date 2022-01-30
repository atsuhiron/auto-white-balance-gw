import sys

import matplotlib.pyplot as plt

import awb


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError:
        print("Input image path.")
        sys.exit(1)

    img, awb_img, param = awb.awb(path, bgr2rgb=True)
    print(param)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(awb_img)
    plt.show()