"""
多线程版本测试文件 - 用于验证线程架构
这个文件展示了多线程架构的核心思想，可以在没有完整硬件环境的情况下测试
"""
import time
import threading
from queue import Queue, Empty

class MockCamera:
    """模拟摄像头"""
    def read(self):
        return True, f"frame_{time.time()}"

class MockUART:
    """模拟UART通信"""
    def __init__(self):
        self.sent_commands = []
        
    def send_cmd(self, cmd):
        self.sent_commands.append(f"CMD: {cmd}")
        print(f"[UART] 发送命令: {cmd}")
        
    def send_turn(self, direction):
        self.sent_commands.append(f"TURN: {direction}")
        print(f"[UART] 发送转向: {direction}")
        
    def send_pid_control(self, value):
        self.sent_commands.append(f"PID: {value}")
        print(f"[UART] 发送PID控制: {value:.2f}")
        
    def read_response(self):
        return None
        
    def get_queue_status(self):
        return {'queue_size': 0, 'waiting_for_ack': False}
        
    def wait_for_queue_empty(self, timeout=1.0):
        return True
        
    def close(self):
        print("[UART] 连接已关闭")

class ThreadData:
    """线程间通信数据结构"""
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

class VisionControlThread(threading.Thread):
    """视觉处理和PID控制线程（模拟版本）"""
    def __init__(self, thread_data, cap, uart):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.cap = cap
        self.uart = uart
        self.frame_count = 0
        
    def run(self):
        print("✓ 视觉控制线程启动")
        
        # 模拟识别目标数字
        target_digit = 3  # 假设目标数字是3
        with self.thread_data.lock:
            self.thread_data.target_digit = target_digit
        print(f"目标数字为：{target_digit}")
        
        # 发送开始直走指令
        self.thread_data.control_queue.put({'cmd': 300001, 'desc': '开始直走'})
        
        while self.thread_data.program_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                    
                self.frame_count += 1
                
                # 模拟不同的场景
                if self.frame_count % 100 == 0:  # 每100帧模拟一次岔路口
                    self._simulate_crossroad()
                else:
                    self._simulate_line_following()
                    
                time.sleep(0.05)  # 20FPS
                
            except Exception as e:
                print(f"视觉控制线程错误: {e}")
                time.sleep(0.1)
                
    def _simulate_crossroad(self):
        """模拟岔路口检测"""
        print("[视觉] 检测到岔路口")
        
        # 发送停止指令
        self.thread_data.control_queue.put({'cmd': 300002, 'desc': '岔路口停止'})
        
        with self.thread_data.lock:
            self.thread_data.crossroad_detected = True
            target_digit = self.thread_data.target_digit
            
        # 模拟数字识别
        import random
        left_num = random.randint(1, 9)
        right_num = random.randint(1, 9)
        print(f"[视觉] 岔路口数字检测. 左: {left_num}, 右: {right_num}")
        
        # 决策逻辑
        if left_num == target_digit:
            decision = 'LEFT'
        elif right_num == target_digit:
            decision = 'RIGHT'
        else:
            decision = 'STRAIGHT'
            
        with self.thread_data.lock:
            self.thread_data.crossroad_decision = decision
            
        print(f"[视觉] 决策: {decision}")
        
        # 发送控制指令
        if decision == 'LEFT':
            self.thread_data.control_queue.put({'turn': 'LEFT', 'desc': '左转'})
        elif decision == 'RIGHT':
            self.thread_data.control_queue.put({'turn': 'RIGHT', 'desc': '右转'})
        else:
            self.thread_data.control_queue.put({'cmd': 300001, 'desc': '直走'})
            
        # 模拟等待红线
        if decision in ['LEFT', 'RIGHT']:
            time.sleep(0.5)  # 模拟转弯时间
            print("[视觉] 转弯完成，等待红线...")
            with self.thread_data.lock:
                self.thread_data.waiting_for_red_line = True
            time.sleep(1.0)  # 模拟等待红线时间
            print("[视觉] 检测到红线，继续循迹")
            self.thread_data.control_queue.put({'cmd': 300001, 'desc': '检测到红线继续'})
            with self.thread_data.lock:
                self.thread_data.waiting_for_red_line = False
                
        # 重置状态
        with self.thread_data.lock:
            self.thread_data.crossroad_detected = False
            self.thread_data.crossroad_decision = None
            
    def _simulate_line_following(self):
        """模拟正常循迹"""
        # 模拟offset值（随机生成，模拟真实的线位置）
        import random
        offset = random.randint(-40, 40)
        
        # 模拟PID计算（简化版本）
        pid_output = -offset * 0.8  # 简单的比例控制
        
        with self.thread_data.lock:
            self.thread_data.current_offset = offset
            self.thread_data.current_pid_output = pid_output
            
        # 发送PID控制指令
        self.thread_data.control_queue.put({
            'pid_control': pid_output, 
            'desc': f'PID控制 offset:{offset} pid:{pid_output:.1f}'
        })

class CommunicationThread(threading.Thread):
    """通信线程"""
    def __init__(self, thread_data, uart):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.uart = uart
        
    def run(self):
        print("✓ 通信线程启动")
        
        while self.thread_data.program_running:
            try:
                # 处理控制指令队列
                try:
                    command = self.thread_data.control_queue.get(timeout=0.1)
                    self._process_command(command)
                except Empty:
                    pass
                    
                # # 模拟检查下位机回传数据
                # response = self.uart.read_response()
                # if response:
                #     print(f"[通信] 下位机回传: {response}")
                    
                # 更新通信状态
                comm_status = self.uart.get_queue_status()
                with self.thread_data.lock:
                    self.thread_data.uart_status = comm_status
                    
                time.sleep(0.01)  # 100Hz通信频率
                
            except Exception as e:
                print(f"通信线程错误: {e}")
                time.sleep(0.1)
                
    def _process_command(self, command):
        """处理单个命令"""
        if 'cmd' in command:
            self.uart.send_cmd(command['cmd'])
        elif 'turn' in command:
            self.uart.send_turn(command['turn'])
        elif 'pid_control' in command:
            self.uart.send_pid_control(command['pid_control'])
        
        if 'desc' in command:
            print(f"[通信] {command['desc']}")

class DisplayThread(threading.Thread):
    """显示线程（模拟版本）"""
    def __init__(self, thread_data):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        
    def run(self):
        print("✓ 显示线程启动")
        print("按 Ctrl+C 退出程序")
        
        frame_count = 0
        while self.thread_data.program_running:
            try:
                frame_count += 1
                
                # 模拟显示更新
                if frame_count % 50 == 0:  # 每50次循环打印一次状态
                    self._print_status()
                    
                time.sleep(0.033)  # 30FPS显示频率
                
                # 模拟按键检测（10秒后自动退出）
                if frame_count > 300:  # 10秒后退出
                    print("[显示] 测试完成，退出程序")
                    with self.thread_data.lock:
                        self.thread_data.program_running = False
                    break
                    
            except Exception as e:
                print(f"显示线程错误: {e}")
                time.sleep(0.1)
                
    def _print_status(self):
        """打印当前状态"""
        with self.thread_data.lock:
            target_digit = self.thread_data.target_digit
            current_offset = self.thread_data.current_offset
            current_pid_output = self.thread_data.current_pid_output
            crossroad_detected = self.thread_data.crossroad_detected
            crossroad_decision = self.thread_data.crossroad_decision
            waiting_for_red_line = self.thread_data.waiting_for_red_line
            uart_status = self.thread_data.uart_status
            
        status = f"[状态] 目标:{target_digit} "
        
        if waiting_for_red_line:
            status += "等待红线 "
        elif crossroad_detected:
            status += f"岔路口:{crossroad_decision or '决策中'} "
        elif current_offset is not None:
            status += f"循迹 offset:{current_offset} pid:{current_pid_output:.1f} "
        else:
            status += "无红线 "
            
        status += f"通信队列:{uart_status['queue_size']}"
        print(status)

def main():
    """主函数"""
    print("=== 多线程架构测试 ===")
    
    # 创建模拟设备
    cap = MockCamera()
    uart = MockUART()
    
    # 创建线程间通信数据结构
    thread_data = ThreadData()
    
    # 创建线程
    vision_thread = VisionControlThread(thread_data, cap, uart)
    comm_thread = CommunicationThread(thread_data, uart)
    display_thread = DisplayThread(thread_data)
    
    try:
        # 启动所有线程
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
        vision_thread.join(timeout=2.0)
        comm_thread.join(timeout=2.0)
        
        print("✓ 所有线程已停止")
        
    except KeyboardInterrupt:
        print("程序被用户中断")
        with thread_data.lock:
            thread_data.program_running = False
    finally:
        print("正在关闭系统...")
        uart.close()
        print("✓ 系统已安全关闭")
        
        print("\n=== 发送的命令总结 ===")
        for cmd in uart.sent_commands:
            print(f"  {cmd}")

if __name__ == "__main__":
    main()
