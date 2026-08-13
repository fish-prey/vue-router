import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# ========== 核心修复：配置Matplotlib中文显示 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签（Windows系统）
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

# 读取图像 - 以灰度模式读取
original_img = cv2.imread("./lena.jpg", 0)
if original_img is None:
    raise FileNotFoundError("请确保 lena.jpg 文件在当前目录下！")


noisy_img = original_img 

# ========== 2. 傅里叶变换 ==========
img_float32 = np.float32(noisy_img)
dft_np = np.fft.fft2(img_float32)
dft_shifted = np.fft.fftshift(dft_np)

magnitude = np.abs(dft_shifted)  # NumPy直接用abs计算复数幅度，更简洁
magnitude_noisy = np.log(magnitude + 1)
magnitude_noisy = cv2.normalize(magnitude_noisy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ========== 3. 频域阈值去噪 ==========
# 计算频域系数的幅度（用于确定阈值
threshold = np.percentile(magnitude, 95)  # 保留95%的主要频率分量，滤除5%的高频噪声
mask = (magnitude > threshold).astype(np.float32)
dft_filtered = dft_shifted * mask  # 直接对复数数组相乘，无需分离实虚部（NumPy优势）

# ========== 4. 逆傅里叶变换 ==========
f_ishift = np.fft.ifftshift(dft_filtered)
img_back_complex = np.fft.ifft2(f_ishift)
img_denoised = np.abs(img_back_complex)
# 归一化到0-255
img_denoised = cv2.normalize(img_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ========== 5. 可视化相关结果 ==========
# 滤波后的幅度谱
magnitude_filtered = np.abs(dft_filtered)
magnitude_filtered = np.log(magnitude_filtered + 1)
magnitude_filtered = cv2.normalize(magnitude_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ========== 6. 显示所有结果 ==========
plt.figure(figsize=(18, 12))

# 2. 含噪图像
plt.subplot(2, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title('含噪声的图像')
plt.axis('off')

# 3. 去噪后的图像
plt.subplot(2, 3, 2)
plt.imshow(img_denoised, cmap='gray')
plt.title('频域阈值去噪后的图像')
plt.axis('off')

# 4. 含噪图像的频域幅度谱 
plt.subplot(2, 3, 4)
plt.imshow(magnitude_noisy, cmap='gray')
plt.title('含噪图像的频域幅度谱')
plt.axis('off')

# 5. 掩膜的频域幅度谱
plt.subplot(2, 3, 5)
plt.imshow(mask, cmap='gray')
plt.title('掩膜的频域幅度谱')
plt.axis('off')

# 6. 滤波后的频域幅度谱 
plt.subplot(2, 3, 6)
plt.imshow(magnitude_filtered, cmap='gray')
plt.title('滤波后的频域幅度谱')
plt.axis('off')

plt.tight_layout()
plt.show()

# 1. 计算含噪图像 vs 原始图像的PSNR/SSIM
psnr_noisy = psnr(original_img, noisy_img, data_range=255)
ssim_noisy = ssim(original_img, noisy_img, data_range=255)
# 2. 计算去噪后图像 vs 原始图像的PSNR/SSIM
psnr_denoised = psnr(original_img, img_denoised, data_range=255)
ssim_denoised = ssim(original_img, img_denoised, data_range=255)

# 打印量化结果
print("="*50)
print("图像去噪效果量化评估")
print("="*50)
print(f"含噪图像 vs 原始图像：")
print(f"  PSNR = {psnr_noisy:.2f} dB")
print(f"  SSIM = {ssim_noisy:.4f}")
print(f"\n去噪后图像 vs 原始图像：")
print(f"  PSNR = {psnr_denoised:.2f} dB")
print(f"  SSIM = {ssim_denoised:.4f}")
print("="*50)