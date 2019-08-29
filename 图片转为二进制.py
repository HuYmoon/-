from PIL import Image


def transform():
    """
    图片转为二进制，输出TXT文档
    :return:
    """
    im = Image.open('手写.png')
    width = im.size[0]
    hight = im.size[1]
    fh = open('1.txt', 'w')
    for i in range(hight):
        for j in range(width):
            color = im.getpixel((j, i))
            colorsum = color[0] + color[1] + color[2]
            if colorsum == 0:
                fh.write('1')
            else:
                fh.write('0')
        fh.write('\n')
    fh.close()



