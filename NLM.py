import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_kernel(sim_rad):
    """生成相似窗口的权重核（均匀核简化版）"""
    kernel = np.zeros((2 * sim_rad + 1, 2 * sim_rad + 1))
    for d in range(1, sim_rad + 1):
        value = 1 / (2 * d + 1) ** 2
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                kernel[sim_rad - i, sim_rad - j] += value  # Python索引从0开始
    kernel = kernel / np.sum(kernel)  # 归一化
    return kernel

def nl_means_filter(img, search_rad, sim_rad, h):
    """
    基础NLM滤波
    :param img: 输入灰度图（uint8/float64）
    :param search_rad: 搜索窗口半径（t）
    :param sim_rad: 相似窗口半径（f）
    :param h: 滤波参数
    :return: 去噪后的图像（uint8）
    """
    img = np.float64(img)
    m, n = img.shape
    pad_size = sim_rad  # 边缘填充大小（对称模式）
    img_pad = np.pad(img, pad_size, mode='symmetric')

    # 生成相似块权重核
    kernel = make_kernel(sim_rad)
    h_sq = h ** 2  # h的平方，避免重复计算

    output = np.zeros_like(img)

    for i in range(m):
        for j in range(n):
            # 当前像素在填充图像中的位置
            i_pad = i + pad_size
            j_pad = j + pad_size
            # 当前像素的相似块（sim_rad×sim_rad）
            current_block = img_pad[i_pad - sim_rad : i_pad + sim_rad + 1,
                                    j_pad - sim_rad : j_pad + sim_rad + 1]

            # 搜索窗口的范围（填充图像中）
            search_min_i = max(i_pad - search_rad, sim_rad)
            search_max_i = min(i_pad + search_rad, m + pad_size - 1)
            search_min_j = max(j_pad - search_rad, sim_rad)
            search_max_j = min(j_pad + search_rad, n + pad_size - 1)

            weight_sum = 0.0
            pixel_sum = 0.0
            max_weight = 0.0

            for x in range(search_min_i, search_max_i + 1):
                for y in range(search_min_j, search_max_j + 1):
                    if x == i_pad and y == j_pad:
                        weight = 1
                        continue  # 跳过自身
                    # 邻居的相似块
                    neighbor_block = img_pad[x - sim_rad : x + sim_rad + 1,
                                             y - sim_rad : y + sim_rad + 1]
                    # 计算块距离（高斯加权欧氏距离）
                    diff = current_block - neighbor_block
                    distance = np.sum(kernel * diff * diff)
                    # 计算权重
                    weight = np.exp(-distance / h_sq)

                    if weight > max_weight:
                        max_weight = weight  # 记录最大权重（用于自身像素）
                    weight_sum += weight
                    pixel_sum += weight * img_pad[x, y]

            # 加上自身像素的权重（最大权重）
            pixel_sum += max_weight * img_pad[i_pad, j_pad]
            weight_sum += max_weight

            # 归一化输出
            output[i, j] = pixel_sum / weight_sum if weight_sum > 0 else img[i, j]

    #return np.uint8(np.clip(output, 0, 255))
    return np.clip(output, 0, 255).astype(np.uint8)

# 读取图像（灰度图）
img = cv2.imread('lenna.jpg', 0)
# 添加高斯噪声（σ=10）
sigma = 10
noisy_img = img + sigma * np.random.randn(*img.shape)
noisy_img = np.uint8(np.clip(noisy_img, 0, 255))

# NLM去噪参数
search_rad = 5  # 搜索窗口半径（11×11）
sim_rad = 2     # 相似窗口半径（5×5）
h = sigma       # 滤波参数

# 去噪
denoised_img = nl_means_filter(noisy_img, search_rad, sim_rad, h)

# 显示结果
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(noisy_img, cmap='gray'), plt.title('Noisy (σ=10)')
plt.subplot(133), plt.imshow(denoised_img, cmap='gray'), plt.title('Denoised (NLM)')
plt.show()

plt.figure(figsize=(5, 8))
plt.imshow(denoised_img-img, cmap='gray')
plt.show()