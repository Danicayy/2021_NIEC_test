"""
数字检测和识别模块
包含所有与数字框选、分割、识别相关的功能
"""

import cv2
import time
import numpy as np
from collections import Counter
from recognize_digit import recognize_digit_from_img

# 设置numpy错误处理，避免溢出警告
np.seterr(over='ignore', invalid='ignore')


class DigitDetector:
    """数字检测器类，负责数字的检测、分割和识别"""
    
    def __init__(self, img_width, img_height, img_size, device):
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = img_size
        self.device = device
        
    def recognize_crossroad_digits(self, model, frame):
        """识别岔路口左右两侧的数字 - 从整个图片中找数字再区分左右"""
        print("开始从整个图片中识别数字...")
        
        try:
            # 从整个图片中找到所有数字
            all_digits = self.find_all_digits_in_frame(model, frame)
            
            if not all_digits:
                print("整个图片中未找到任何数字")
                return None, None, None, None
            
            print(f"找到 {len(all_digits)} 个数字")
            
            # 根据位置区分左右
            frame_center_x = self.img_width // 2
            left_digits = []
            right_digits = []
            
            for digit_info in all_digits:
                digit_value, box, confidence = digit_info
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                
                print(f"数字 {digit_value} 位置: ({center_x}, {(y1+y2)//2}), 置信度: {confidence:.3f}")
                
                # 根据中心点位置区分左右
                if center_x < frame_center_x:
                    left_digits.append(digit_info)
                else:
                    right_digits.append(digit_info)
            
            # 选择最可能的左右数字（选择置信度最高的）
            left_num = None
            right_num = None
            left_box = None
            right_box = None
            
            if left_digits:
                # 按置信度排序，选择最高的
                left_digits.sort(key=lambda x: x[2], reverse=True)
                left_num = left_digits[0][0]
                left_box = left_digits[0][1]
                print(f"左侧选择数字: {left_num} (置信度: {left_digits[0][2]:.3f})")
                
            if right_digits:
                # 按置信度排序，选择最高的
                right_digits.sort(key=lambda x: x[2], reverse=True)
                right_num = right_digits[0][0]
                right_box = right_digits[0][1]
                print(f"右侧选择数字: {right_num} (置信度: {right_digits[0][2]:.3f})")
            
            print(f"最终识别结果: 左={left_num}, 右={right_num}")
            return left_num, right_num, left_box, right_box
            
        except Exception as e:
            print(f"识别过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

    def find_all_digits_in_frame(self, model, frame):
        """从整个帧中找到所有数字"""
        print("扫描整个图片寻找数字...")
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 找到轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_candidates = []
        
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 过滤条件：合理的大小和宽高比
            min_area = 100  # 最小面积
            max_area = self.img_width * self.img_height * 0.5  # 最大面积不超过图像的10%
            min_aspect = 0.2  # 最小宽高比
            max_aspect = 3.0  # 最大宽高比
            
            aspect_ratio = w / h if h > 0 else 0
            
            if (min_area < area < max_area and 
                min_aspect < aspect_ratio < max_aspect and
                w > 10 and h > 15):  # 最小尺寸限制
                
                # 添加边距
                margin = max(5, min(w, h) // 10)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(self.img_width, x + w + margin)
                y2 = min(self.img_height, y + h + margin)
                
                roi_box = (x1, y1, x2, y2)
                
                # 使用模型识别这个候选区域
                try:
                    digit_value = recognize_digit_from_img(model, frame, roi_box,
                                                         device=self.device,
                                                         img_size=self.img_size,
                                                         debug=False)
                    
                    if digit_value is not None:
                        # 计算置信度（这里简化为基于区域特征的评分）
                        confidence = self.calculate_digit_confidence(frame, roi_box, area, aspect_ratio)
                        digit_candidates.append((digit_value, roi_box, confidence))
                        print(f"在位置 ({x1},{y1},{x2},{y2}) 找到数字 {digit_value}, 置信度: {confidence:.3f}")
                        
                except Exception as e:
                    # 识别失败，跳过
                    continue
        
        print(f"总共找到 {len(digit_candidates)} 个有效数字")
        return digit_candidates
    
    def calculate_digit_confidence(self, frame, roi_box, area, aspect_ratio):
        """计算数字识别的置信度"""
        x1, y1, x2, y2 = roi_box
        
        # 基础分数
        base_score = 0.5
        
        # 面积评分 (适中的面积得分更高)
        ideal_area = 800  # 理想面积
        area_score = 1.0 - abs(area - ideal_area) / ideal_area
        area_score = max(0, min(1, area_score))
        
        # 宽高比评分 (接近0.6的宽高比得分更高，适合数字)
        ideal_aspect = 0.6
        aspect_score = 1.0 - abs(aspect_ratio - ideal_aspect) / ideal_aspect
        aspect_score = max(0, min(1, aspect_score))
        
        # 位置评分 (图片中央区域得分更高)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        frame_center_x = self.img_width / 2
        frame_center_y = self.img_height / 2
        
        # 距离中心越近得分越高
        distance_from_center = ((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)**0.5
        max_distance = (self.img_width**2 + self.img_height**2)**0.5 / 2
        position_score = 1.0 - distance_from_center / max_distance
        
        # 综合评分
        confidence = (base_score * 0.3 + 
                     area_score * 0.3 + 
                     aspect_score * 0.2 + 
                     position_score * 0.2)
        
        return confidence

    def apply_simple_segmentation(self, gray):
        """简单但有效的分割方法"""
        # 高斯模糊去噪
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Otsu自动阈值
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作去除小噪声
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return thresh

    def apply_contrast_enhancement(self, gray):
        """对比度增强后的分割"""
        # CLAHE (对比度限制自适应直方图均衡)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 自适应阈值 - 针对160x120分辨率
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 10)
        
        return thresh

    def apply_multi_scale_segmentation(self, gray):
        """多尺度分割方法 - 结合不同尺度的特征"""
        # 多个尺度的高斯模糊
        blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
        blur2 = cv2.GaussianBlur(gray, (7, 7), 0)
        blur3 = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # 多个阈值的Otsu
        _, thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 投票机制：多数决定
        combined = (thresh1.astype(np.float32) + thresh2.astype(np.float32) + thresh3.astype(np.float32)) / 3
        result = (combined > 127).astype(np.uint8) * 255
        
        # 精细化处理
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return result

    def apply_region_growing_segmentation(self, gray):
        """基于区域生长的分割方法 - 使用OpenCV实现"""
        # 预处理
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 使用多个阈值创建种子点
        _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 自适应阈值作为补充 - 针对160x120分辨率
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 8)
        
        # 结合两种阈值结果
        combined = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        
        # 使用OpenCV的分水岭算法
        # 距离变换
        dist_transform = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
        
        # 寻找局部最大值
        _, markers = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        markers = np.uint8(markers)
        
        # 寻找连通组件作为标记
        _, markers = cv2.connectedComponents(markers)
        
        # 应用分水岭
        markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
        
        # 创建结果mask
        result = np.zeros_like(gray)
        result[markers > 1] = 255  # 排除背景(0)和边界(-1)
        
        # 选择最大的连通区域
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            return mask
        
        return combined

    def apply_gradient_based_segmentation(self, gray):
        """基于梯度的分割方法 - 更适合数字边缘检测"""
        # 预处理
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 计算梯度
        grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
        
        # 梯度方向的自适应阈值
        _, thresh_grad = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 结合原图的强度信息
        _, thresh_intensity = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 组合梯度和强度信息
        combined = cv2.bitwise_and(thresh_grad, thresh_intensity)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=1)
        combined = cv2.erode(combined, kernel, iterations=1)
        
        # 填充空洞
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return combined

    def apply_adaptive_histogram_segmentation(self, gray):
        """自适应直方图分割 - 根据局部特征自动调整参数"""
        # 计算局部统计信息
        h, w = gray.shape
        block_size = min(h, w) // 4
        
        # 分块处理
        result = np.zeros_like(gray)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # 提取块
                block = gray[i:min(i+block_size, h), j:min(j+block_size, w)]
                
                if block.size == 0:
                    continue
                    
                # 计算块的统计特征
                mean_val = float(np.mean(block))
                std_val = float(np.std(block))
                
                # 根据标准差选择分割策略
                if std_val > 30:  # 高对比度区域
                    _, block_thresh = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:  # 低对比度区域
                    # 使用更敏感的自适应阈值
                    block_thresh = cv2.adaptiveThreshold(block, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 
                                                       min(11, block.shape[0]//2*2-1), 5)
                
                result[i:min(i+block_size, h), j:min(j+block_size, w)] = block_thresh
        
        # 全局优化
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return result

    def smart_segmentation_selector(self, gray, debug=False):
        """智能分割方法选择器 - 根据图像特征自动选择最佳分割方法"""
        # 分析图像特征
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        # 计算对比度（使用Michelson对比度）
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        
        # 安全的对比度计算，避免溢出
        if max_val == min_val:
            contrast = 0.0
        else:
            # 使用更安全的计算方式
            numerator = max_val - min_val
            denominator = max_val + min_val
            if denominator > 0 and numerator >= 0:
                contrast = numerator / denominator
            else:
                contrast = 0.0
        
        # 计算边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)
        
        if debug:
            print(f"图像分析:")
            print(f"  平均值: {mean_val:.1f}")
            print(f"  标准差: {std_val:.1f}")
            print(f"  对比度: {contrast:.3f}")
            print(f"  边缘密度: {edge_density:.3f}")
        
        # 根据特征选择分割方法
        if contrast > 0.3 and std_val > 40:
            # 高对比度，使用简单分割
            method_name = "Simple Segmentation (High Contrast)"
            result = self.apply_simple_segmentation(gray)
        elif std_val < 20:
            # 低对比度，使用对比度增强
            method_name = "Contrast Enhancement (Low Contrast)"
            result = self.apply_contrast_enhancement(gray)
        elif edge_density > 0.1:
            # 边缘丰富，使用梯度分割
            method_name = "Gradient Based (Rich Edges)"
            result = self.apply_gradient_based_segmentation(gray)
        elif mean_val < 100 or mean_val > 180:
            # 偏暗或偏亮，使用自适应直方图
            method_name = "Adaptive Histogram (Extreme Brightness)"
            result = self.apply_adaptive_histogram_segmentation(gray)
        else:
            #默认使用多尺度分割
            method_name = "Multi-Scale (Default)"
            result = self.apply_multi_scale_segmentation(gray)
        
        if debug:
            print(f"  选择方法: {method_name}")
        
        return result, method_name

    def find_digit_boxes(self, frame, use_smart_segmentation=True):
        """在帧中查找数字候选框"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 根据模式选择分割方法
        # if use_smart_segmentation:
        thresh, method_name = self.smart_segmentation_selector(gray, debug=False)
        # else:
        #     # 默认使用简单有效的分割
        #     thresh = self.apply_simple_segmentation(gray)
        #     method_name = "Simple Segmentation"
        
        # 查找外部轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / (w + 1e-6)
            area = w * h
            # 仅保留形状、面积符合数字的框 - 针对160x120分辨率
            if 20 < w < 100 and 30 < h < 160 and 1.0 < aspect_ratio < 3.5 and 500 < area < 7000:
                candidate_boxes.append((x, y, x+w, y+h))
    # 按靠近中心排序
        candidate_boxes.sort(key=lambda box: abs((box[0]+box[2])//2 - self.img_width//2))
        return candidate_boxes, method_name
            # x, y, w, h = cv2.boundingRect(cnt)
            # if 10 < w < 80 and 20 < h < 100:
            #     cx = x + w // 2
            #     cy = y + h // 2
            #     w *= 1.2
            #     h *= 1.2
            #     x1 = max(0, int(cx - w // 2))
            #     y1 = max(0, int(cy - h // 2))
            #     x2 = min(self.img_width, int(cx + w // 2))
            #     y2 = min(self.img_height, int(cy + h // 2))
            #     candidate_boxes.append((x1, y1, x2, y2))
        
        # return candidate_boxes, method_name

    def recognize_digits_in_boxes(self, model, frame, boxes):
        """识别候选框中的数字"""
        recognized_digits = []
        
        for box in boxes:
            num = recognize_digit_from_img(model, frame, box, 
                                         device=self.device, 
                                         img_size=self.img_size, 
                                         debug=False)
            if num is not None:
                recognized_digits.append(num)
        
        return recognized_digits

    def recognize_target_digit(self, cap, model):
        """3秒自动确认数字"""
        print("请将目标数字放到摄像头前，系统将自动识别（3秒后确认）")
        
        numbers = []
        start_time = time.time()
        display_time = 3  # 3秒自动确认
        confirmed = False

        while not confirmed:
            ret, frame = cap.read()
            if not ret:
                continue

            # 使用简单但有效的分割方法
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display = frame.copy()
            frame_numbers = []

            # 绘制所有检测到的轮廓框
            candidate_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 15 < w < 100 and 25 < h < 120:  # 针对160x120分辨率的大小范围
                    # 扩展框
                    cx = x + w // 2
                    cy = y + h // 2
                    w_ = int(w * 1.3)
                    h_ = int(h * 1.3)
                    x1 = max(0, cx - w_ // 2)
                    y1 = max(0, cy - h_ // 2)
                    x2 = min(self.img_width, cx + w_ // 2)
                    y2 = min(self.img_height, cy + h_ // 2)
                    
                    candidate_boxes.append((x1, y1, x2, y2))
                    
                    # 绘制候选框（黄色）
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(display, f"W:{w} H:{h}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 识别候选框中的数字
            for i, (x1, y1, x2, y2) in enumerate(candidate_boxes):
                try:
                    num = recognize_digit_from_img(model, frame, (x1, y1, x2, y2), 
                                                device=self.device, 
                                                img_size=self.img_size, 
                                                debug=False)
                    if num is not None:
                        frame_numbers.append(num)
                        # 绘制识别成功的框（绿色）和数字
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(display, f"{num}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                        cv2.putText(display, f"ID:{i}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)
                    else:
                        # 绘制识别失败的框（红色）
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display, "?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                except Exception as e:
                    # 绘制异常的框（红色虚线效果）
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(display, "ERR", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                    continue
            
            # 实时累积3秒内所有帧检测到的数字
            numbers.extend(frame_numbers)

            # 显示统计信息和倒计时
            elapsed = time.time() - start_time
            remain = max(0, int(display_time - elapsed) + 1)
            cv2.putText(display, f"Candidates: {len(candidate_boxes)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if numbers:
                most_common = Counter(numbers).most_common(1)[0][0]
                cv2.putText(display, f"Detected: {most_common}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No digit detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, f"Auto confirm in {remain}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 绘制图像中心线和参考区域
            center_x = self.img_width // 2
            center_y = self.img_height // 2
            cv2.line(display, (center_x, 0), (center_x, self.img_height), (255, 0, 0), 1)
            cv2.line(display, (0, center_y), (self.img_width, center_y), (255, 0, 0), 1)
            cv2.circle(display, (center_x, center_y), 50, (255, 0, 0), 2)
            
            cv2.imshow("Target Digit Recognition", display)
            cv2.waitKey(30)

            if elapsed > display_time:
                if numbers:
                    most_common = Counter(numbers).most_common(1)[0][0]
                    print(f"自动识别结果: {most_common}")
                    confirmed = True
                else:
                    print("3秒内未检测到数字，请重试。")
                    cv2.destroyWindow("Target Digit Recognition")
                    return None

        cv2.destroyWindow("Target Digit Recognition")
        return most_common

    # def recognize_target_digit_fast(self, cap, model):
    #     """快速识别目标数字 - 带框选显示"""
    #     print("快速识别模式: 3秒内最常识别的数字")
        
    #     frame_count = 0
    #     recognition_results = []
        
    #     while frame_count < 30:  # 约3秒(每帧100ms)
    #         ret, frame = cap.read()
    #         if not ret:
    #             continue
                
    #         frame_count += 1
    #         display = frame.copy()
            
    #         # 快速分割和检测
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #         _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
    #         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
    #         best_box = None
    #         best_digit = None
            
    #         # 找到最合适的候选框
    #         for contour in contours:
    #             x, y, w, h = cv2.boundingRect(contour)
    #             if 20 < w < 80 and 30 < h < 100:  # 针对160x120分辨率的数字大小范围
    #                 aspect_ratio = h / (w + 1e-6)
    #                 if 1.0 < aspect_ratio < 2.5:  # 数字长宽比
    #                     # 扩展框
    #                     cx = x + w // 2
    #                     cy = y + h // 2
    #                     new_w = int(w * 1.2)
    #                     new_h = int(h * 1.2)
    #                     x1 = max(0, cx - new_w // 2)
    #                     y1 = max(0, cy - new_h // 2)
    #                     x2 = min(self.img_width, cx + new_w // 2)
    #                     y2 = min(self.img_height, cy + new_h // 2)
                        
    #                     # 绘制候选框
    #                     cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
    #                     # 尝试识别
    #                     try:
    #                         digit = recognize_digit_from_img(model, frame, (x1, y1, x2, y2), 
    #                                                        device=self.device, 
    #                                                        img_size=self.img_size, 
    #                                                        debug=False)
    #                         if digit is not None:
    #                             best_box = (x1, y1, x2, y2)
    #                             best_digit = digit
    #                             recognition_results.append(digit)
                                
    #                             # 绘制成功识别的框（绿色）
    #                             cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #                             cv2.putText(display, f"{digit}", (x1, y1 - 10), 
    #                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    #                             break
    #                     except:
    #                         # 绘制识别失败的框（红色）
    #                         cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #                         continue
            
    #         # 显示进度和结果
    #         progress = int((frame_count / 30) * 100)
    #         cv2.putText(display, f"Progress: {progress}%", (10, 30), 
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
    #         if recognition_results:
    #             from collections import Counter
    #             most_common = Counter(recognition_results).most_common(1)[0]
    #             cv2.putText(display, f"Current: {most_common[0]} ({most_common[1]}x)", 
    #                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
    #         cv2.imshow("Fast Recognition", display)
            
    #         # 允许提前退出
    #         key = cv2.waitKey(100) & 0xFF
    #         if key == 27:  # ESC退出
    #             break
    #         elif key == 32:  # 空格提前确认
    #             break
        
    #     cv2.destroyWindow("Fast Recognition")
        
    #     if recognition_results:
    #         from collections import Counter
    #         final_digit = Counter(recognition_results).most_common(1)[0][0]
    #         print(f"快速识别结果: {final_digit}")
    #         return final_digit
    #     else:
    #         print("快速识别失败")
    #         return None
        # """识别岔路口左右两侧的数字"""
        # left_box = (int(self.img_width*0.05), int(self.img_height*0.2), 
        #            int(self.img_width*0.3), int(self.img_height*0.7))
        # right_box = (int(self.img_width*0.7), int(self.img_height*0.2), 
        #             int(self.img_width*0.95), int(self.img_height*0.7))
        
        # left_num = recognize_digit_from_img(model, frame, left_box, 
        #                                   device=self.device, 
        #                                   img_size=self.img_size, 
        #                                   debug=False)
        # right_num = recognize_digit_from_img(model, frame, right_box, 
        #                                    device=self.device, 
        #                                    img_size=self.img_size, 
        #                                    debug=False)
        
        # return left_num, right_num, left_box, right_box

    def evaluate_segmentation_quality(self, original, segmented, debug=False):
        """评估分割质量"""
        # 计算分割覆盖率
        white_pixels = cv2.countNonZero(segmented)
        total_pixels = segmented.shape[0] * segmented.shape[1]
        coverage = float(white_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        
        # 计算连通组件数量
        num_labels, labels = cv2.connectedComponents(segmented)
        
        # 计算最大连通组件占比
        if num_labels > 1:
            component_sizes = []
            for i in range(1, num_labels):
                component_size = int(np.sum(labels == i))
                component_sizes.append(component_size)
            
            largest_component = max(component_sizes) if component_sizes else 0
            largest_ratio = float(largest_component) / float(white_pixels) if white_pixels > 0 else 0.0
        else:
            largest_ratio = 0
        
        # 计算边缘一致性
        original_edges = cv2.Canny(original, 50, 150)
        segmented_edges = cv2.Canny(segmented, 50, 150)
        
        # 安全的边缘一致性计算
        original_edge_sum = float(np.sum(original_edges > 0))
        if original_edge_sum > 0:
            edge_consistency = float(np.sum(cv2.bitwise_and(original_edges, segmented_edges) > 0)) / original_edge_sum
        else:
            edge_consistency = 0.0
        
        # 综合评分 (0-1之间)
        quality_score = (
            (0.3 * (1 - abs(coverage - 0.2))) +  # 理想覆盖率约20%
            (0.4 * largest_ratio) +              # 最大连通组件占比
            (0.3 * edge_consistency)             # 边缘一致性
        )
        
        if debug:
            print(f"分割质量评估:")
            print(f"  覆盖率: {coverage:.3f}")
            print(f"  连通组件数: {num_labels-1}")
            print(f"  最大组件占比: {largest_ratio:.3f}")
            print(f"  边缘一致性: {edge_consistency:.3f}")
            print(f"  综合评分: {quality_score:.3f}")
        
        return quality_score
