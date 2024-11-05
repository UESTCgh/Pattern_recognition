import cv2
import numpy as np

# 图片显示函数
def showImage(name:str,image):
    """ 显示图片的函数 """
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    kernel = np.ones((4, 4), np.uint8) # 8
    kernel2 = np.ones((1, 1), np.uint8)  # 8
    image = cv2.imread('315.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    showImage('blur',blurred)
    edge = cv2.Canny(blurred, 20, 60)  # 用Canny算子提取边缘 转化为二值图像 50 70

    edge=cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel)
    # edge=cv2.erode(edge, kernel2)
    showImage('edge',edge)
    cv2.imwrite("./results/edge.jpg", edge)


    contour = image.copy()
    (cnts, _) = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cv2.drawContours(contour, cnts, -1, (0, 255, 0), 2)  # 绘制轮廓
    showImage('contour',contour)
    cv2.imwrite("./results/contour.jpg", contour)


    count = 0  #
    margin = 5  # 裁剪边距
    draw_rect = image.copy()
    draw_mask=image.copy()
    for i, contour in enumerate(cnts):
        arc = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)  # 计算包围形状的面积
        if area < 500 or arc <200:  # 过滤面积小于15的形状
            continue
        count += 1
        rect = cv2.minAreaRect(contour)  # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        box = np.int0(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
        cv2.drawContours(draw_rect, [box], 0, (255, 0, 0), 2)  # 绘制轮廓最小外接矩形

        # 创建一个空白掩模
        mask = np.zeros_like(draw_mask)
        white_background = np.zeros_like(draw_mask)
        # 在掩模上绘制小鱼的轮廓区域
        cv2.drawContours(mask, [contour], -1, (255, 255, 255),
                         thickness=-1)  # 255,255,255 represents white color
        # 使用掩模将小鱼区域从原始彩色图像中提取出来
        fish_extracted = cv2.bitwise_and(draw_mask, mask)
        # 将小鱼区域复制到白色背景上
        white_background = cv2.bitwise_or(white_background, fish_extracted)
        # xiaoyu=cv2.drawContours(white_background, [contour], -1, (0, 0, 0),
        #                  thickness=cv2.FILLED)  # 0,0,0 represents black color
        showImage('xiaoyu',white_background)
        cv2.imwrite("./koutu/{}.jpg".format(count),white_background)

        h, w = image.shape[:2]  # 原图像的高和宽
        rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1  # 最小外接矩形的宽和高
        if rect_w <= rect_h:
            x, y = int(box[1][0]), int(box[1][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
            rotated_image = cv2.warpAffine(image, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_h + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_w + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        else:
            x, y = int(box[2][0]), int(box[2][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2] + 90, 1)
            rotated_image = cv2.warpAffine(image, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_w + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_h + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        print("rice #{}".format(count))
        # cv2.imshow("rotated_canvas", rotated_canvas)
        cv2.imwrite("./rotation-results/{}.jpg".format(count), rotated_canvas)
        cv2.waitKey(0)
    showImage('draw_rect',draw_rect)
    cv2.imwrite("./results/rect.jpg", draw_rect)