"""
数字识别调试模块
包含所有调试、可视化、测试相关的功能
"""

import cv2
import time
import numpy as np
import os
from recognize_digit import recognize_digit_from_img


class DigitDebugger:
    """数字识别调试器类"""
    
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
    
    def debug_segmentation_methods(self, gray):
        """显示不同分割方法的效果对比
        按 'q' 退出调试模式，按 '1'-'7' 选择特定方法
        """
        print("分割方法调试模式:")
        print("1 - 简单分割  2 - 对比度增强  3 - 自适应阈值")
        print("4 - 多尺度分割  5 - 区域生长  6 - 梯度分割  7 - 自适应直方图")
        print("按数字键查看单个方法，按 'a' 查看所有方法对比，按 'q' 退出")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('a'):
                # 显示所有方法对比
                self.show_all_segmentation_methods(gray)
            elif key >= ord('1') and key <= ord('7'):
                # 显示单个方法
                method_idx = key - ord('1')
                self.show_single_segmentation_method(gray, method_idx)

    def show_all_segmentation_methods(self, gray):
        """显示所有分割方法的对比"""
        from digit_detector import DigitDetector
        detector = DigitDetector(self.img_width, self.img_height, 16, 'cpu')
        
        # 应用不同的分割方法
        methods = [
            ("Simple", detector.apply_simple_segmentation(gray)),
            ("Enhanced", detector.apply_contrast_enhancement(gray)),
            ("Adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 15, 8)),
            ("Multi-Scale", detector.apply_multi_scale_segmentation(gray)),
            ("Region Growing", detector.apply_region_growing_segmentation(gray)),
            ("Gradient", detector.apply_gradient_based_segmentation(gray)),
            ("Adaptive Hist", detector.apply_adaptive_histogram_segmentation(gray))
        ]
        
        # 创建对比图像 (3x3网格)
        h, w = gray.shape
        comparison = np.zeros((h*3, w*3), dtype=np.uint8)
        
        # 原图放在左上角
        comparison[0:h, 0:w] = gray
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 其他方法按顺序排列
        positions = [
            (0, 1), (0, 2),  # 第一行剩余位置
            (1, 0), (1, 1), (1, 2),  # 第二行
            (2, 0), (2, 1)   # 第三行前两个
        ]
        
        for i, ((name, result), (row, col)) in enumerate(zip(methods, positions)):
            y_start, y_end = row*h, (row+1)*h
            x_start, x_end = col*w, (col+1)*w
            comparison[y_start:y_end, x_start:x_end] = result
            cv2.putText(comparison, name, (x_start+10, y_start+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imshow("All Segmentation Methods", comparison)

    def show_single_segmentation_method(self, gray, method_idx):
        """显示单个分割方法的详细结果"""
        from digit_detector import DigitDetector
        detector = DigitDetector(self.img_width, self.img_height, 16, 'cpu')
        
        methods = [
            ("Simple Segmentation", detector.apply_simple_segmentation),
            ("Contrast Enhancement", detector.apply_contrast_enhancement),
            ("Adaptive Threshold", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                                  cv2.THRESH_BINARY_INV, 15, 8)),
            ("Multi-Scale", detector.apply_multi_scale_segmentation),
            ("Region Growing", detector.apply_region_growing_segmentation),
            ("Gradient Based", detector.apply_gradient_based_segmentation),
            ("Adaptive Histogram", detector.apply_adaptive_histogram_segmentation)
        ]
        
        if method_idx < len(methods):
            name, method_func = methods[method_idx]
            result = method_func(gray)
            
            # 创建对比显示
            h, w = gray.shape
            comparison = np.zeros((h, w*2), dtype=np.uint8)
            comparison[0:h, 0:w] = gray
            comparison[0:h, w:2*w] = result
            
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(comparison, name, (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # 显示统计信息
            white_pixels = cv2.countNonZero(result)
            total_pixels = result.shape[0] * result.shape[1]
            coverage = (white_pixels / total_pixels) * 100
            
            cv2.putText(comparison, f"Coverage: {coverage:.1f}%", (w+10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.imshow(f"Method: {name}", comparison)

    def debug_digit_recognition(self, cap, model, device, img_size):
        """数字识别调试模式 - 实时显示预处理步骤
        按 'q' 退出，按 's' 保存当前帧用于分析
        """
        print("数字识别调试模式:")
        print("- 按 'q' 退出调试")
        print("- 按 's' 保存当前识别结果")
        print("- 按 'd' 开启/关闭详细调试信息")
        
        debug_detail = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            display = frame.copy()
            
            # 定义多个测试区域
            test_boxes = [
                (int(self.img_width*0.05), int(self.img_height*0.2), int(self.img_width*0.3), int(self.img_height*0.7)),  # 左侧
                (int(self.img_width*0.7), int(self.img_height*0.2), int(self.img_width*0.95), int(self.img_height*0.7)),   # 右侧
                (int(self.img_width*0.35), int(self.img_height*0.3), int(self.img_width*0.65), int(self.img_height*0.6))   # 中央
            ]
            
            colors = [(0,255,0), (255,0,0), (0,255,255)]
            positions = ["Left", "Right", "Center"]
            
            for i, (box, color, pos) in enumerate(zip(test_boxes, colors, positions)):
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # 识别数字
                digit = recognize_digit_from_img(model, frame, box, device=device, 
                                               img_size=img_size, debug=debug_detail)
                
                # 显示结果
                result_text = f"{pos}: {digit if digit is not None else 'None'}"
                cv2.putText(display, result_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 显示调试信息
            cv2.putText(display, "Debug Mode - Press 'q':quit, 's':save, 'd':detail", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            cv2.imshow("Digit Recognition Debug", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Digit Recognition Debug")
                break
            elif key == ord('s'):
                # 保存当前帧和识别结果
                filename = f"debug_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display)
                print(f"保存调试帧: {filename}")
                frame_count += 1
            elif key == ord('d'):
                debug_detail = not debug_detail
                print(f"详细调试信息: {'开启' if debug_detail else '关闭'}")

    def validate_model_input(self, model, device, img_size, test_image_path=None):
        """验证模型输入格式是否正确"""
        print("验证模型输入格式...")
        
        if test_image_path and os.path.exists(test_image_path):
            # 使用提供的测试图像
            test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 创建一个简单的测试图像（数字"7"的简单模拟）
            test_img = np.zeros((32, 32), dtype=np.uint8)
            cv2.line(test_img, (5, 5), (25, 5), 255, 2)   # 上横线
            cv2.line(test_img, (25, 5), (15, 25), 255, 2)  # 斜线
        
        # 预处理
        from recognize_digit import preprocess_digit_roi
        processed = preprocess_digit_roi(test_img, img_size, debug=True)
        
        # 转换为张量
        from torchvision import transforms
        import torch
        tensor_img = transforms.ToTensor()(processed).unsqueeze(0).to(device)
        
        print(f"张量形状: {tensor_img.shape}")
        print(f"张量数据类型: {tensor_img.dtype}")
        print(f"张量值范围: [{tensor_img.min():.3f}, {tensor_img.max():.3f}]")
        
        # 测试模型
        with torch.no_grad():
            output = model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            
            print(f"模型输出形状: {output.shape}")
            print("各类别概率:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  数字 {i}: {prob.item():.4f}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def analyze_frame_quality(self, frame):
        """分析帧质量和光照条件"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 基本统计信息
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 对比度分析
        min_val = np.min(gray)
        max_val = np.max(gray)
        contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        # 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 噪声估计（使用拉普拉斯变换）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        quality_info = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'noise_level': laplacian_var,
            'quality_score': self._calculate_quality_score(mean_brightness, contrast, edge_density, laplacian_var)
        }
        
        return quality_info
    
    def _calculate_quality_score(self, brightness, contrast, edge_density, noise_level):
        """计算综合质量评分"""
        # 亮度评分 (理想范围 100-180)
        brightness_score = 1.0 - abs(brightness - 140) / 140
        brightness_score = max(0, brightness_score)
        
        # 对比度评分 (越高越好，但不超过1)
        contrast_score = min(contrast * 2, 1.0)
        
        # 边缘密度评分 (适中为好)
        edge_score = 1.0 - abs(edge_density - 0.05) / 0.05
        edge_score = max(0, edge_score)
        
        # 噪声评分 (越低越好)
        noise_score = max(0, 1.0 - noise_level / 1000)
        
        # 综合评分
        quality_score = (brightness_score * 0.3 + 
                        contrast_score * 0.3 + 
                        edge_score * 0.2 + 
                        noise_score * 0.2)
        
        return quality_score

    def draw_quality_info(self, frame, quality_info):
        """在帧上绘制质量信息"""
        display = frame.copy()
        
        # 绘制质量信息
        y_offset = 30
        line_height = 25
        
        info_texts = [
            f"Brightness: {quality_info['mean_brightness']:.1f}",
            f"Contrast: {quality_info['contrast']:.3f}",
            f"Edge Density: {quality_info['edge_density']:.3f}",
            f"Noise Level: {quality_info['noise_level']:.1f}",
            f"Quality Score: {quality_info['quality_score']:.3f}"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = y_offset + i * line_height
            cv2.putText(display, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 质量等级颜色
        quality_score = quality_info['quality_score']
        if quality_score > 0.7:
            quality_color = (0, 255, 0)  # 绿色 - 好
            quality_text = "GOOD"
        elif quality_score > 0.4:
            quality_color = (0, 255, 255)  # 黄色 - 一般
            quality_text = "FAIR"
        else:
            quality_color = (0, 0, 255)  # 红色 - 差
            quality_text = "POOR"
        
        cv2.putText(display, f"Quality: {quality_text}", (10, y_offset + 5 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
        
        return display

    def create_test_patterns(self):
        """创建测试图案用于验证识别效果"""
        test_patterns = []
        
        # 创建数字0-9的简单测试图案
        for digit in range(10):
            pattern = np.zeros((64, 64), dtype=np.uint8)
            
            if digit == 0:
                cv2.rectangle(pattern, (20, 15), (44, 49), 255, 2)
            elif digit == 1:
                cv2.line(pattern, (32, 15), (32, 49), 255, 2)
            elif digit == 2:
                cv2.line(pattern, (20, 15), (44, 15), 255, 2)
                cv2.line(pattern, (44, 15), (44, 32), 255, 2)
                cv2.line(pattern, (44, 32), (20, 32), 255, 2)
                cv2.line(pattern, (20, 32), (20, 49), 255, 2)
                cv2.line(pattern, (20, 49), (44, 49), 255, 2)
            elif digit == 3:
                cv2.line(pattern, (20, 15), (44, 15), 255, 2)
                cv2.line(pattern, (44, 15), (44, 49), 255, 2)
                cv2.line(pattern, (20, 32), (44, 32), 255, 2)
                cv2.line(pattern, (20, 49), (44, 49), 255, 2)
            # ... 可以继续添加其他数字的图案
            
            test_patterns.append((digit, pattern))
        
        return test_patterns

    def test_recognition_accuracy(self, model, device, img_size):
        """测试识别准确率"""
        test_patterns = self.create_test_patterns()
        correct_count = 0
        total_count = len(test_patterns)
        
        print("开始测试识别准确率...")
        
        for true_digit, pattern in test_patterns:
            # 模拟在帧中的位置
            test_frame = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            
            # 将图案放置在帧中心
            y_start = (self.img_height - pattern.shape[0]) // 2
            x_start = (self.img_width - pattern.shape[1]) // 2
            
            test_frame[y_start:y_start+pattern.shape[0], 
                      x_start:x_start+pattern.shape[1], :] = pattern[:,:,np.newaxis]
            
            # 定义识别区域
            box = (x_start, y_start, x_start+pattern.shape[1], y_start+pattern.shape[0])
            
            # 识别数字
            recognized_digit = recognize_digit_from_img(model, test_frame, box, 
                                                     device=device, img_size=img_size, debug=False)
            
            if recognized_digit == true_digit:
                correct_count += 1
                print(f"✓ 数字 {true_digit}: 识别正确")
            else:
                print(f"✗ 数字 {true_digit}: 识别为 {recognized_digit}")
        
        accuracy = correct_count / total_count * 100
        print(f"\n识别准确率: {accuracy:.1f}% ({correct_count}/{total_count})")
        
        return accuracy
