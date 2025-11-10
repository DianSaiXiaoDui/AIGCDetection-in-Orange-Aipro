import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.stats import entropy


def safe_cv2_convert(image):
    """
    安全的图像格式转换
    """
    if isinstance(image, Image.Image):
        # PIL Image 转 OpenCV
        img_array = np.array(image)
        
        # 确保是数值数组
        if img_array.dtype == np.object_:
            # 处理特殊情况：重新转换为正确的格式
            img_array = np.array(image.convert('RGB'))
        
        # 确保是3通道RGB
        if len(img_array.shape) == 2:  # 灰度图
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # RGB转BGR（OpenCV格式）
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    elif isinstance(image, np.ndarray):
        # 已经是numpy数组
        img_cv = image.copy()
        if len(img_cv.shape) == 2:  # 灰度
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:  # 可能是RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    else:
        raise ValueError(f"不支持的图像格式: {type(image)}")
    
    return img_cv

def saliency_based_crop(image, target_size=256):
    """
    基于显著度检测找到信息最丰富的区域
    """
    # 转换为OpenCV格式
    #img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv = safe_cv2_convert(image)
    
    # 方法1: 使用OpenCV的显著性检测
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(img_cv)
    
    if not success:
        # 方法2: 使用基于对比度的简单显著度
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        saliency_map = cv2.Laplacian(gray, cv2.CV_64F)
        saliency_map = np.abs(saliency_map)
    
    # 归一化显著度图
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # 找到显著度最高的区域
    height, width = image.size
    crop_height, crop_width = target_size, target_size
    
    # 滑动窗口计算平均显著度
    best_score = -1
    best_box = (0, 0, crop_width, crop_height)
    
    for y in range(0, height - crop_height, crop_height // 4):
        for x in range(0, width - crop_width, crop_width // 4):
            region_score = np.mean(saliency_map[y:y+crop_height, x:x+crop_width])
            if region_score > best_score:
                best_score = region_score
                best_box = (x, y, x+crop_width, y+crop_height)
    
    # 裁剪并保存
    cropped = image.crop(best_box)
    cropped.save("region_saliency.png")
    #print(f"🎯 显著度裁剪: 位置{best_box}, 显著度得分: {best_score:.3f}")
    
    return cropped
    
def edge_density_crop(image, target_size=256):
    """
    基于边缘密度找到细节最丰富的区域
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 计算边缘密度
    height, width = image.size
    crop_size = target_size
    
    best_density = -1
    best_box = (0, 0, crop_size, crop_size)
    
    for y in range(0, height - crop_size, crop_size // 4):
        for x in range(0, width - crop_size, crop_size // 4):
            region_edges = edges[y:y+crop_size, x:x+crop_size]
            edge_density = np.sum(region_edges > 0) / (crop_size * crop_size)
            
            if edge_density > best_density:
                best_density = edge_density
                best_box = (x, y, x+crop_size, y+crop_size)
    
    cropped = image.crop(best_box)
    cropped.save("region_edges.png")
    #print(f"🔍 边缘密度裁剪: 位置{best_box}, 边缘密度: {best_density:.3f}")
    
    return cropped  
    

def entropy_based_crop(image, target_size=256):
    """
    基于信息熵找到信息量最大的区域
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    height, width = image.size
    crop_size = target_size
    
    best_entropy = -1
    best_box = (0, 0, crop_size, crop_size)
    
    for y in range(0, height - crop_size, crop_size // 4):
        for x in range(0, width - crop_size, crop_size // 4):
            region = gray[y:y+crop_size, x:x+crop_size]
            
            # 计算区域的信息熵
            hist = np.histogram(region, bins=256, range=(0, 255))[0]
            hist = hist / hist.sum()  # 归一化
            region_entropy = entropy(hist[hist > 0])  # 避免log(0)
            
            if region_entropy > best_entropy:
                best_entropy = region_entropy
                best_box = (x, y, x+crop_size, y+crop_size)
    
    cropped = image.crop(best_box)
    cropped.save("region_entropy.png")
    print(f"📊 信息熵裁剪: 位置{best_box}, 信息熵: {best_entropy:.3f}")
    
    return cropped   