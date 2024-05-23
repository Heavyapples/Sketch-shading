import numpy as np
import cv2
import scipy.sparse
import scipy.sparse.linalg


class ColorizationUsingOptimization():
    def __init__(self, ori_img, skt_img):
        self.ori_img = cv2.imread(ori_img).astype(np.float32) / 255
        self.skt_img = cv2.imread(skt_img).astype(np.float32) / 255

        # Ensure the images have the same size
        assert self.ori_img.shape[:2] == self.skt_img.shape[
                                         :2], "The original image and sketch image must have the same size"

        # rgb2yuv
        self.ori_yuv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2YUV)
        self.skt_yuv = cv2.cvtColor(self.skt_img, cv2.COLOR_BGR2YUV)

        # 交换U和V通道
        self.skt_yuv[:, :, [1, 2]] = self.skt_yuv[:, :, [2, 1]]

        # get image size
        self.height, self.width = self.ori_img.shape[:2]
        self.img_size = self.width * self.height
        # separate y u v
        self.y_ori = self.ori_yuv[:, :, 0]
        self.u_skt = self.skt_yuv[:, :, 1].reshape(self.img_size)
        self.v_skt = self.skt_yuv[:, :, 2].reshape(self.img_size)
        # get sketched pixels position
        self.get_skt_pos()
        # build weight matrix and b
        self.build_weight_b()
        # solve WX = b1 and colorization
        self.color()

    def get_skt_pos(self):
        assert self.ori_img.shape == self.skt_img.shape, "not the same image size"
        self.skt_pos = np.zeros((self.ori_img.shape[0], self.ori_img.shape[1]))
        for i in range(self.skt_pos.shape[0]):
            for j in range(self.skt_pos.shape[1]):
                if (self.ori_img[i][j][0] != self.skt_img[i][j][0]):
                    self.skt_pos[i][j] = 1

    def build_weight_b(self):
        weight_data = []
        row_inds = []
        col_inds = []
        # construct weight matrix
        for h in range(self.height):
            for w in range(self.width):
                if self.skt_pos[h][w] == 0:
                    neighbor_value = []
                    for i in range(h - 1, h + 2):
                        for j in range(w - 1, w + 2):
                            if (0 <= i and i < self.height - 1 and 0 <= j and j < self.width - 1):
                                if (h != i) | (w != j):
                                    neighbor_value.append(self.y_ori[h, w])
                                    row_inds.append(h * self.width + w)
                                    col_inds.append(i * self.width + j)
                    sigma = np.var(np.append(neighbor_value, self.y_ori[h, w]))
                    if sigma < 1e-6:
                        sigma = 1e-6
                    w_rs = np.exp(- np.power(neighbor_value - self.y_ori[h][w], 2) / sigma)
                    w_rs = - w_rs / np.sum(w_rs)
                    for item in w_rs:
                        weight_data.append(item)
                weight_data.append(1)
                row_inds.append(h * self.width + w)
                col_inds.append(h * self.width + w)
        self.W = scipy.sparse.csc_matrix((weight_data, (row_inds, col_inds)), shape=(self.img_size, self.img_size))
        # construct b
        self.b_u = np.zeros(self.img_size)
        self.b_v = np.zeros(self.img_size)
        # skt_pos_vec is the indix of nonzero element
        skt_pos_vec = np.nonzero(self.skt_pos.reshape(self.img_size))
        self.b_u[skt_pos_vec] = self.u_skt[skt_pos_vec]
        self.b_v[skt_pos_vec] = self.v_skt[skt_pos_vec]

    def color(self):
        u_res = scipy.sparse.linalg.spsolve(self.W, self.b_u).reshape((self.height, self.width))
        v_res = scipy.sparse.linalg.spsolve(self.W, self.b_v).reshape((self.height, self.width))

        yuv_res = np.dstack((self.y_ori.astype(np.float32), u_res.astype(np.float32), v_res.astype(np.float32)))
        self.rgb_res = cv2.cvtColor(yuv_res, cv2.COLOR_YUV2RGB)

if __name__ == "__main__":
    # 实际文件名
    ori_img = "1_after.png"
    skt_img = "1_skt.png"

    colorization = ColorizationUsingOptimization(ori_img, skt_img)
    # 保存和显示结果图像
    cv2.imwrite("output_colorized_image.png", (colorization.rgb_res * 255).clip(0, 255).astype(np.uint8))
    # 读取原始灰度图像、草图和上色后的图像
    original_gray_image = cv2.imread(ori_img)
    sketch_image = cv2.imread(skt_img)
    colorized_image = (colorization.rgb_res * 255).clip(0, 255).astype(np.uint8)

    # 将三张图片水平堆叠在一起
    combined_image = cv2.hconcat([original_gray_image, sketch_image, colorized_image])

    # 显示合并后的图像
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()