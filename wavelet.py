import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def wavelet_denoise(image, wavelet='db4', level=2, threshold_mode='soft'):
    """
    小波变换图像去噪核心函数
    :param image: 输入灰度图像（二维numpy数组）
    :param wavelet: 小波基函数（db4通用性最强）
    :param level: 小波分解层数
    :param threshold_mode: 阈值模式（soft软阈值/ hard硬阈值）
    :return: 去噪后的图像
    """
    # 1. 二维离散小波分解 (cA: 低频分量, (cH, cV, cD): 水平/垂直/对角高频分量)
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    
    # 2. 计算通用阈值（小波去噪标准阈值）
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745  # 噪声标准差估计
    threshold = sigma * np.sqrt(2 * np.log(image.size))
    
    # 3. 对所有高频分量进行阈值去噪
    coeffs_thresh = [coeffs[0]]  # 保留低频分量
    for i in range(1, len(coeffs)):
        # 对高频分量逐通道阈值处理
        cH_thresh = pywt.threshold(coeffs[i][0], value=threshold, mode=threshold_mode)
        cV_thresh = pywt.threshold(coeffs[i][1], value=threshold, mode=threshold_mode)
        cD_thresh = pywt.threshold(coeffs[i][2], value=threshold, mode=threshold_mode)
        coeffs_thresh.append((cH_thresh, cV_thresh, cD_thresh))
    
    # 4. 小波逆变换，重构去噪图像
    denoised_img = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    
    # 裁剪边界（小波变换会产生微小边界偏移）
    denoised_img = denoised_img[:image.shape[0], :image.shape[1]]
    
    # 归一化到0-255并转换为8位图像
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
    return denoised_img

def add_gaussian_noise(image, mean=0, var=10):
    """
    为图像添加高斯噪声
    :param image: 原始图像
    :param mean: 噪声均值
    :param var: 噪声方差（越大噪声越强）
    :return: 带噪图像
    """
    image = image.astype(np.float32)
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_img = image + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 读取图像
    img_path = "./lenna.jpg" 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 添加高斯噪声
    noisy_img = add_gaussian_noise(img, mean=0, var=20)
    
    # 小波去噪
    denoised_img = wavelet_denoise(
        image=noisy_img,
        wavelet='db4',    # 小波基：db4/haar/sym4
        level=3,          # 分解层数：1-3最佳
        threshold_mode='soft'  # 软阈值去噪效果更平滑
    )
    
    # 结果可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('原始图像'), plt.axis('off')
    plt.subplot(132), plt.imshow(noisy_img, cmap='gray'), plt.title('带噪图像'), plt.axis('off')
    plt.subplot(133), plt.imshow(denoised_img, cmap='gray'), plt.title('小波去噪后图像'), plt.axis('off')
    plt.tight_layout()
    plt.show()
