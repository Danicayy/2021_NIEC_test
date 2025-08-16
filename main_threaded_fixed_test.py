"""
修复后的多线程架构测试
解决了摄像头资源冲突问题
"""
import time
import threading
from queue import Queue, Empty

class MockCamera:
    """模拟摄像头"""
    def __init__(self):
        self.frame_count = 0
        
    def read(self):
        self.frame_count += 1
        return True, f"frame_{self.frame_count}_{time.time():.3f}"
    
    def set(self, prop, value):
        pass

class ThreadData:
    def __init__(self):
        self.lock = threading.Lock()
        self.control_queue = Queue()
        self.target_digit = None
        self.current_offset = None
        self.current_pid_output = None
        self.crossroad_detected = False
        self.crossroad_decision = None
        self.program_running = True
        self.waiting_for_red_line = False
        self.uart_status = {'queue_size': 0, 'waiting_for_ack': False}
        # 新增：共享帧数据
        self.current_frame = None
        self.frame_ready = False

class CameraThread(threading.Thread):
    """摄像头读取线程 - 唯一访问摄像头的线程"""
    def __init__(self, thread_data, cap):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.cap = cap
        
    def run(self):
        print("✓ 摄像头线程启动")
        
        while self.thread_data.program_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # 将帧数据共享给其他线程
                    with self.thread_data.lock:
                        self.thread_data.current_frame = frame  # 模拟环境不需要copy
                        self.thread_data.frame_ready = True
                else:
                    print("⚠️ 摄像头读取失败")
                    with self.thread_data.lock:
                        self.thread_data.frame_ready = False
                    
                time.sleep(0.033)  # 30FPS读取频率
                
            except Exception as e:
                print(f"摄像头线程错误: {e}")
                with self.thread_data.lock:
                    self.thread_data.frame_ready = False
                time.sleep(0.1)

class VisionControlThread(threading.Thread):
    """视觉处理和PID控制线程（修复版本）"""
    def __init__(self, thread_data, cap_for_target_recognition):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.cap_for_target = cap_for_target_recognition
        self.frame_count = 0
        
    def run(self):
        print("✓ 视觉控制线程启动")
        
        # 1. 模拟识别目标数字（使用独立摄像头访问）
        target_digit = 3
        with self.thread_data.lock:
            self.thread_data.target_digit = target_digit
        print(f"目标数字为：{target_digit}")
        
        # 2. 发送开始直走指令
        self.thread_data.control_queue.put({'cmd': 300001, 'desc': '开始直走'})
        
        # 3. 主循环：从共享内存读取帧
        while self.thread_data.program_running:
            try:
                # 从共享内存获取帧
                frame = self._get_current_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                self.frame_count += 1
                
                # 模拟处理逻辑
                if self.frame_count % 100 == 0:  # 每100帧模拟一次岔路口
                    self._simulate_crossroad(frame)
                else:
                    self._simulate_line_following(frame)
                    
                time.sleep(0.05)  # 20FPS处理频率
                
            except Exception as e:
                print(f"视觉控制线程错误: {e}")
                time.sleep(0.1)
    
    def _get_current_frame(self):
        """从共享内存获取当前帧"""
        with self.thread_data.lock:
            if self.thread_data.frame_ready and self.thread_data.current_frame is not None:
                return self.thread_data.current_frame  # 在模拟环境中直接返回
        return None
                
    def _simulate_crossroad(self, frame):
        """模拟岔路口检测"""
        print(f"[视觉] 处理帧: {frame} - 检测到岔路口")
        # 其余逻辑与之前相同...
        self.thread_data.control_queue.put({'cmd': 300002, 'desc': '岔路口停止'})
        
    def _simulate_line_following(self, frame):
        """模拟正常循迹"""
        # 模拟PID计算
        import random
        offset = random.randint(-40, 40)
        pid_output = -offset * 0.8
        
        with self.thread_data.lock:
            self.thread_data.current_offset = offset
            self.thread_data.current_pid_output = pid_output
            
        # 发送PID控制指令
        self.thread_data.control_queue.put({
            'pid_control': pid_output, 
            'desc': f'PID控制 frame:{frame} offset:{offset} pid:{pid_output:.1f}'
        })

class CommunicationThread(threading.Thread):
    """通信线程"""
    def __init__(self, thread_data):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        
    def run(self):
        print("✓ 通信线程启动")
        
        while self.thread_data.program_running:
            try:
                try:
                    command = self.thread_data.control_queue.get(timeout=0.1)
                    print(f"[通信] 处理命令: {command['desc']}")
                except Empty:
                    pass
                    
                time.sleep(0.01)  # 100Hz通信频率
                
            except Exception as e:
                print(f"通信线程错误: {e}")
                time.sleep(0.1)

class DisplayThread(threading.Thread):
    """显示线程（修复版本）"""
    def __init__(self, thread_data):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        
    def run(self):
        print("✓ 显示线程启动")
        print("测试将在10秒后自动退出...")
        
        frame_count = 0
        while self.thread_data.program_running:
            try:
                # 从共享内存获取帧
                frame = self._get_current_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                frame_count += 1
                
                # 模拟显示处理
                if frame_count % 30 == 0:  # 每30帧显示一次状态
                    print(f"[显示] 处理帧: {frame}")
                    
                time.sleep(0.033)  # 30FPS显示频率
                
                # 10秒后自动退出
                if frame_count > 300:
                    print("[显示] 测试完成，退出程序")
                    with self.thread_data.lock:
                        self.thread_data.program_running = False
                    break
                    
            except Exception as e:
                print(f"显示线程错误: {e}")
                time.sleep(0.1)
    
    def _get_current_frame(self):
        """从共享内存获取当前帧"""
        with self.thread_data.lock:
            if self.thread_data.frame_ready and self.thread_data.current_frame is not None:
                return self.thread_data.current_frame
        return None

def main():
    """主函数"""
    print("=== 修复后的多线程架构测试 ===")
    print("解决了摄像头资源冲突问题")
    
    # 创建模拟设备
    cap = MockCamera()
    
    print("正在初始化摄像头...")
    time.sleep(0.1)
    print("✓ 摄像头初始化成功")
    
    # 创建线程间通信数据结构
    thread_data = ThreadData()
    
    # 创建线程
    camera_thread = CameraThread(thread_data, cap)
    vision_thread = VisionControlThread(thread_data, cap)  # 目标识别用的独立访问
    comm_thread = CommunicationThread(thread_data)
    display_thread = DisplayThread(thread_data)
    
    try:
        # 先启动摄像头线程
        camera_thread.start()
        time.sleep(0.5)  # 等待摄像头初始化
        
        # 再启动其他线程
        vision_thread.start()
        comm_thread.start()
        display_thread.start()
        
        print("✓ 所有线程已启动")
        
        # 等待显示线程结束
        display_thread.join()
        
        print("正在停止所有线程...")
        with thread_data.lock:
            thread_data.program_running = False
            
        # 等待其他线程结束
        camera_thread.join(timeout=2.0)
        vision_thread.join(timeout=2.0)
        comm_thread.join(timeout=2.0)
        
        print("✓ 所有线程已停止")
        
    except KeyboardInterrupt:
        print("程序被用户中断")
        with thread_data.lock:
            thread_data.program_running = False
    finally:
        print("✓ 系统已安全关闭")

if __name__ == "__main__":
    main()
