import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

# 可调参数 - 针对160x120分辨率优化
CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值，可以调整为 0.5-0.8
CLAHE_CLIP_LIMIT = 3.0     # 对比度增强限制，可以调整为 2.0-5.0
BINARY_BLOCK_SIZE = 11     # 自适应阈值块大小，160x120分辨率使用11
BINARY_C_VALUE = 2         # 自适应阈值常数，160x120分辨率使用2

# def multi_stage_resize(img, target_size, stages=(64, 32, 16)):
#     result = img.copy()
#     for s in stages:
#         if result.shape[0] > s or result.shape[1] > s:
#             # 只在最后一次缩放后模糊
#             result = cv2.resize(result, (s, s), interpolation=cv2.INTER_LINEAR)
#     # 最后只模糊一次，且用很小的sigma
#     result = cv2.GaussianBlur(result, (3, 3), 0.2)
#     if result.shape[0] != target_size or result.shape[1] != target_size:
#         result = cv2.resize(result, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
#     return result

def recognize_digit_from_img(model, frame, roi_box, device='cpu', img_size=16, debug=False, use_enhanced=True):
    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]  # 从原图中提取感兴趣区域
    
    # 检查ROI是否有效
    if roi.size == 0:
        return None
    
    # 转换为灰度图像
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi
    
    # 选择预处理方法
    if use_enhanced:
        processed_roi = apply_enhanced_preprocessing(gray_roi, img_size, debug)
    else:
        processed_roi = preprocess_digit_roi(gray_roi, img_size, debug)
    
    # 转换为张量
    tensor_img = transforms.ToTensor()(processed_roi).unsqueeze(0).to(device)
    
    # 归一化到[0,1]范围（如果模型需要的话）
    # tensor_img = tensor_img / 255.0  # 根据模型训练时的归一化方式调整
    
    with torch.no_grad():
        output = model(tensor_img)
        # 获取概率分布
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
        # 设置置信度阈值，避免低置信度的错误识别
        if confidence.item() < CONFIDENCE_THRESHOLD:  # 可以调整这个阈值
            return None
            
        if debug:
            print(f"Predicted: {pred.item()}, Confidence: {confidence.item():.3f}")
            # 显示所有类别的概率
            for i, prob in enumerate(probabilities[0]):
                print(f"  Class {i}: {prob.item():.3f}")
        
        return int(pred.item())

def preprocess_digit_roi(gray_roi, img_size=16, debug=False):
    """
    改进的数字ROI预处理函数（缩放前不要做二值化！）
    """
    # 1. 增强对比度
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    
    # 2. 高斯模糊降噪
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 3. 找到最大连通区域，裁剪ROI（同原逻辑）
    _, temp_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(temp_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        margin = max(2, min(w, h) // 10)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray_roi.shape[1] - x, w + 2 * margin)
        h = min(gray_roi.shape[0] - y, h + 2 * margin)
        digit_roi = blurred[y:y+h, x:x+w]
    else:
        digit_roi = blurred

    # 4. 保持宽高比居中，填充为正方形
    h2, w2 = digit_roi.shape
    max_dim = max(h2, w2)
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
    start_x = (max_dim - w2) // 2
    start_y = (max_dim - h2) // 2
    square_img[start_y:start_y+h2, start_x:start_x+w2] = digit_roi

    # 5. 先缩放到目标尺寸（用INTER_LINEAR或INTER_LANCZOS4，能抗锯齿）
    resized = cv2.resize(square_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 6. 再做自适应阈值或Otsu二值化
    binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, BINARY_BLOCK_SIZE, BINARY_C_VALUE)
    # 或
    # _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. 形态学操作去噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8. 平滑边缘（可选，轻度高斯模糊）
    # binary = cv2.GaussianBlur(binary, (3, 3), 0)

    # 9. 自动反转（确保数字是白色）
    mean_val = np.mean(binary)
    if mean_val > 127:
        binary = cv2.bitwise_not(binary)

    if debug:
        cv2.imshow("ROI Gray", gray_roi)
        cv2.imshow("Enhanced", enhanced)
        cv2.imshow("Resized", resized)
        cv2.imshow("Final Binary", binary)
        cv2.waitKey(1)

    return Image.fromarray(binary)

def apply_enhanced_preprocessing(gray_roi, img_size=16, debug=False):
    """
    增强的预处理方法 - 针对数字识别优化
    """
    # 1. 图像质量评估
    mean_val = np.mean(gray_roi)
    std_val = np.std(gray_roi)
    
    # 2. 根据图像特征选择预处理策略
    if std_val < 15:  # 低对比度
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray_roi)
        
        # 使用更敏感的自适应阈值 - 针对160x120分辨率
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 9, 3)
    elif mean_val < 80 or mean_val > 180:  # 极端亮度
        # 使用直方图均衡化
        enhanced = cv2.equalizeHist(gray_roi)
        
        # 使用Otsu阈值
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # 正常情况
        # 轻微模糊去噪
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        
        # 标准自适应阈值
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, BINARY_BLOCK_SIZE, BINARY_C_VALUE)
    
    # 3. 确保数字是白色，背景是黑色
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # 4. 寻找并提取主要数字区域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最合适的轮廓（基于面积和形状）
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            
            # 过滤条件：合理的面积和宽高比
            if (area > binary.size * 0.02 and  # 至少占总面积的2%
                area < binary.size * 0.8 and   # 不超过总面积的80%
                0.2 < aspect_ratio < 3.0):     # 合理的宽高比
                valid_contours.append((area, cnt))
        
        if valid_contours:
            # 选择面积最大的有效轮廓
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            largest_contour = valid_contours[0][1]
            
            # 创建mask，只保留最大连通组件
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            binary = cv2.bitwise_and(binary, mask)
            
            # 获取紧密边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 添加适当的边距
            margin = max(2, min(w, h) // 8)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(binary.shape[1] - x, w + 2 * margin)
            h = min(binary.shape[0] - y, h + 2 * margin)
            
            # 裁剪到数字区域
            digit_roi = binary[y:y+h, x:x+w]
            
            # 5. 创建正方形图像并居中
            max_dim = max(w, h)
            # 确保尺寸是偶数，便于居中
            if max_dim % 2 == 1:
                max_dim += 1
                
            square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # 计算居中位置
            start_x = (max_dim - w) // 2
            start_y = (max_dim - h) // 2
            square_img[start_y:start_y+h, start_x:start_x+w] = digit_roi
            
            # 6. 调整到目标尺寸
            resized = cv2.resize(square_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        else:
            # 没有找到有效轮廓，直接调整尺寸
            resized = cv2.resize(binary, (img_size, img_size), interpolation=cv2.INTER_AREA)
    else:
        # 没有找到轮廓，直接调整尺寸
        resized = cv2.resize(binary, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # 7. 最后的优化：边缘平滑
    # 应用轻微的高斯模糊来平滑边缘
    resized = cv2.GaussianBlur(resized, (3, 3), 0.5)
    # 重新二值化
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imshow("Original ROI", gray_roi)
        cv2.imshow("Enhanced", enhanced if 'enhanced' in locals() else gray_roi)
        cv2.imshow("Binary", binary)
        cv2.imshow("Final Processed", resized)
        print(f"预处理统计: 均值={mean_val:.1f}, 标准差={std_val:.1f}")
        cv2.waitKey(1)
    
    # 转换为PIL Image
    return Image.fromarray(resized)