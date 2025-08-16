import cv2
import time
import threading
from queue import Queue, Empty
from collections import deque

from digit_model import load_digit_model
from digit_detector import DigitDetector
from digit_debug import DigitDebugger
from line_follower import LineFollower
from uart_sender import UartSender
from crossroad_detector import is_crossroad
from pid_controller import PIDController
from pid_config import get_preset, list_presets

IMG_WIDTH = 160
IMG_HEIGHT = 120
IMG_SIZE = 16
DEVICE = 'cpu'
MODEL_PATH = 'simple_digit_cnn.pth'
SERIAL_PORT = '/dev/serial0'
BAUDRATE = 9600
PID_OUTPUT_LIMIT = 80

def draw_and_save_digit_boxes(img, img_path, digit_detector, model, output_dir="crossroad_images_debug"):
    boxes, method_name = digit_detector.find_digit_boxes(img)
    debug_img = img.copy()
    img_results = []
    for i, box in enumerate(boxes):
        num = digit_detector.recognize_digits_in_boxes(model, img, [box])
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        if num and len(num) > 0:
            img_results.append((num[0], center_x))
        color = (0, 255, 0) if num and len(num) > 0 else (0, 0, 255)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        if num and len(num) > 0:
            cv2.putText(debug_img, f"{num[0]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(debug_img, "?", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(debug_img, f"#{i+1}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"debug_{basename}")
    cv2.imwrite(save_path, debug_img)
    print(f"[DEBUG] 已保存数字框可视化：{save_path}")
    return img_results

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
        self.current_frame = None
        self.frame_ready = False
        self.is_returning = False
        self.pause_control = False

class CameraThread(threading.Thread):
    def __init__(self, thread_data, cap, save_dir='crossroad_images'):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.cap = cap
        self.save_dir = save_dir
        self.saved_images = deque(maxlen=10)
        import os
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.last_save_time = 0
        self.image_count = 0

    def run(self):
        print("✓ 摄像头线程启动")
        while self.thread_data.program_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.thread_data.lock:
                        self.thread_data.current_frame = frame.copy()
                        self.thread_data.frame_ready = True
                    now = time.time()
                    if now - self.last_save_time >= 0.008:
                        import os
                        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                        ms = int((now - int(now)) * 1000)
                        img_name = f"{timestamp}_{ms:03d}.png"
                        img_path = os.path.join(self.save_dir, img_name)
                        try:
                            cv2.imwrite(img_path, frame)
                            self.saved_images.append(img_path)
                            self.last_save_time = now
                            self.image_count += 1
                            if self.image_count % 10 == 0:
                                print(f"已保存 {self.image_count} 张图像，当前队列长度: {len(self.saved_images)}")
                        except Exception as e:
                            print(f"保存图像失败 {img_path}: {e}")
                else:
                    print("⚠️ 摄像头读取失败")
                    with self.thread_data.lock:
                        self.thread_data.frame_ready = False
                time.sleep(0.001)
            except Exception as e:
                print(f"摄像头线程错误: {e}")
                with self.thread_data.lock:
                    self.thread_data.frame_ready = False

class VisionControlThread(threading.Thread):
    def __init__(self, thread_data,  model, digit_detector, digit_debugger, 
                 line_follower, pid_controller, camera_thread):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.model = model
        self.digit_detector = digit_detector
        self.digit_debugger = digit_debugger
        self.line_follower = line_follower
        self.pid_controller = pid_controller
        self.camera_thread = camera_thread

    def run(self):
        print("✓ 视觉控制线程启动")
        print(f"视觉控制线程启动，目标数字为：{self.thread_data.target_digit}")
        self.thread_data.control_queue.put({'cmd': 300001, 'desc': '开始直走'})
        while self.thread_data.program_running:
            try:
                with self.thread_data.lock:
                    if self.thread_data.pause_control:
                        time.sleep(0.1)
                        continue
                frame = self._get_current_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                if is_crossroad(frame, IMG_WIDTH, IMG_HEIGHT):
                    self._handle_crossroad(frame)
                else:
                    self._handle_line_following(frame)
                time.sleep(0.1)
            except Exception as e:
                print(f"视觉控制线程错误: {e}")

    def _get_current_frame(self):
        with self.thread_data.lock:
            if self.thread_data.frame_ready and self.thread_data.current_frame is not None:
                return self.thread_data.current_frame.copy()
        return None   

    def _handle_crossroad(self, frame):
        with self.thread_data.lock:
            if self.thread_data.pause_control:
                return
            is_returning = self.thread_data.is_returning
        if is_returning:
            print("返程模式：岔路口停车1秒后直走，不做数字识别")
            self.thread_data.control_queue.put({'cmd': 300002, 'desc': '返程岔路口停车'})
            time.sleep(1)
            self.thread_data.control_queue.put({'cmd': 300001, 'desc': '返程直走'})
            with self.thread_data.lock:
                self.thread_data.crossroad_detected = False
            print("返程岔路口处理完成")
            return
        print("检测到岔路口，停止并进行数字识别...")
        self.thread_data.control_queue.put({'cmd': 400001, 'desc': '岔路口发送'})
        self.thread_data.control_queue.put({'cmd': 300002, 'desc': '岔路口停止'})
        with self.thread_data.lock:
            self.thread_data.crossroad_detected = True
            target_digit = self.thread_data.target_digit
        camera_thread = None
        for t in threading.enumerate():
            if isinstance(t, CameraThread):
                camera_thread = t
                break
        if not camera_thread or not camera_thread.saved_images:
            print("没有保存的岔路口图像，无法进行数字识别")
            with self.thread_data.lock:
                self.thread_data.crossroad_detected = False
            return
        imgs = list(camera_thread.saved_images)
        if len(imgs) >= 3:
            selected_imgs = imgs[max(0, len(imgs)-3-10+1):len(imgs)-2]
        else:
            selected_imgs = imgs[:]
        all_left_nums = []
        all_right_nums = []
        print("=" * 50)
        print(f"开始从{len(selected_imgs)}张图像中识别数字（目标数字: {target_digit}）")
        print("=" * 50)
        for i, img_path in enumerate(selected_imgs, 1):
            print(f"[{i:2d}/{len(selected_imgs)}] 正在处理图像: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{i:2d}/{len(selected_imgs)}] 无法读取图像: {img_path}")
                continue
            img_results = draw_and_save_digit_boxes(img, img_path, self.digit_detector, self.model)
            frame_center_x = self.digit_detector.img_width // 2
            for num, center_x in img_results:
                if center_x < frame_center_x:
                    all_left_nums.append(num)
                else:
                    all_right_nums.append(num)
        print("=" * 50)
        print("所有图像处理完成，开始统计结果")
        print("=" * 50)
        from collections import Counter
        left_mode = Counter(all_left_nums).most_common(1)[0][0] if all_left_nums else None
        right_mode = Counter(all_right_nums).most_common(1)[0][0] if all_right_nums else None
        print("=" * 50)
        print("数字识别统计结果:")
        if all_left_nums:
            left_counter = Counter(all_left_nums)
            left_confidence = left_counter[left_mode] / len(all_left_nums)
            print(f"左侧识别投票: {left_counter}")
            print(f"左侧众数: {left_mode} (置信度: {left_confidence:.2f}, {left_counter[left_mode]}/{len(all_left_nums)}框)")
        else:
            print("左侧未识别到任何数字")
        if all_right_nums:
            right_counter = Counter(all_right_nums)
            right_confidence = right_counter[right_mode] / len(all_right_nums)
            print(f"右侧识别投票: {right_counter}")
            print(f"右侧众数: {right_mode} (置信度: {right_confidence:.2f}, {right_counter[right_mode]}/{len(all_right_nums)}框)")
        else:
            print("右侧未识别到任何数字")
        print(f"目标数字: {target_digit}")
        print("=" * 50)
        decision = 'STRAIGHT'
        if left_mode == target_digit and right_mode == target_digit:
            decision = 'LEFT'
        elif left_mode == target_digit:
            decision = 'LEFT'
        elif right_mode == target_digit:
            decision = 'RIGHT'
        else:
            decision = 'STRAIGHT'
        with self.thread_data.lock:
            self.thread_data.crossroad_decision = decision
        self.pid_controller.reset()
        if decision == 'LEFT':
            print(f"发送左转指令...")
            self.thread_data.control_queue.put({'cmd': 200002,'desc': '左转'})
            time.sleep(2)
        elif decision == 'RIGHT':
            print(f"发送右转指令...")
            self.thread_data.control_queue.put({'cmd': 200001, 'desc': '右转'})
            time.sleep(2)
        else:
            print(f"发送直走指令...")
            self.thread_data.control_queue.put({'cmd': 300001, 'desc': '直走'})
        if decision in ['LEFT', 'RIGHT']:
            self._wait_for_red_line_after_turn()
        with self.thread_data.lock:
            self.thread_data.crossroad_detected = False
            self.thread_data.crossroad_decision = None
        print("岔路口处理完成")

    def _wait_for_red_line_after_turn(self):
        print("停车等待检测到红线...")
        with self.thread_data.lock:
            self.thread_data.waiting_for_red_line = True
        self.thread_data.control_queue.put({'cmd': 300002, 'desc': '等待红线停止'})
        while self.thread_data.program_running:
            wait_frame = self._get_current_frame()
            if wait_frame is None:
                time.sleep(0.001)
                continue
            offset, found = self.line_follower.get_offset(wait_frame)
            if found:
                print("检测到红线，继续循迹...")
                self.thread_data.control_queue.put({'cmd': 300001, 'desc': '检测到红线继续'})
                break
        with self.thread_data.lock:
            self.thread_data.waiting_for_red_line = False

    def _handle_line_following(self, frame):
        offset, found = self.line_follower.get_offset(frame)
        with self.thread_data.lock:
            if self.thread_data.pause_control:
                return
        if found:
            pid_output = self.pid_controller.compute(0, offset)
            with self.thread_data.lock:
                self.thread_data.current_offset = offset
                self.thread_data.current_pid_output = pid_output
            self.thread_data.control_queue.put({
                'pid_control': pid_output, 
                'desc': f'PID控制 offset:{offset:.1f} pid:{pid_output:.1f}'
            })
        else:
            with self.thread_data.lock:
                self.thread_data.current_offset = None
                self.thread_data.current_pid_output = None
            self.pid_controller.reset()
            self.thread_data.control_queue.put({'cmd': 300002, 'desc': '丢失红线停止'})

class CommunicationThread(threading.Thread):
    def __init__(self, thread_data, uart):
        super().__init__(daemon=True)
        self.thread_data = thread_data
        self.uart = uart

    def run(self):
        print("✓ 通信线程启动")
        while self.thread_data.program_running:
            try:
                # 监听串口信号
                resp = self.uart.read_response()
                if resp is not None:
                    with self.thread_data.lock:
                        if "before_back" in resp:
                            print("收到 before_back，下位机要求暂停发送控制指令！")
                            self.thread_data.pause_control = True
                        elif "back" in resp:
                            print("收到 back，进入返程巡线模式。")
                            self.thread_data.pause_control = False
                            self.thread_data.is_returning = True
                        elif "weight_right" in resp:
                            # weight_right仅影响主线程启动，不影响这里
                            pass
                with self.thread_data.lock:
                    if self.thread_data.pause_control:
                        time.sleep(0.05)
                        continue
                try:
                    command = self.thread_data.control_queue.get(timeout=0.1)
                    self._process_command(command)
                except Empty:
                    pass
                comm_status = self.uart.get_queue_status()
                with self.thread_data.lock:
                    self.thread_data.uart_status = comm_status
                time.sleep(0.05)
            except Exception as e:
                print(f"通信线程错误: {e}")
                time.sleep(0.1)
    def _process_command(self, command):
        if 'cmd' in command:
            self.uart.send_cmd(command['cmd'])
        elif 'turn' in command:
            self.uart.send_turn(command['turn'])
        elif 'pid_control' in command:
            self.uart.send_pid_control(command['pid_control'])
        if 'desc' in command:
            print(f"通信: {command['desc']}")

class DisplayThread(threading.Thread):
    def __init__(self, thread_data):
        super().__init__(daemon=True)
        self.thread_data = thread_data

    def run(self):
        print("✓ 显示线程启动")
        while self.thread_data.program_running:
            try:
                frame = self._get_current_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                display_frame = self._create_display_frame(frame)
                cv2.imshow('DebugView', display_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    with self.thread_data.lock:
                        self.thread_data.program_running = False
                    break
                time.sleep(0.11)
            except Exception as e:
                print(f"显示线程错误: {e}")
                time.sleep(0.1)
    def _get_current_frame(self):
        with self.thread_data.lock:
            if self.thread_data.frame_ready and self.thread_data.current_frame is not None:
                return self.thread_data.current_frame.copy()
        return None
    def _create_display_frame(self, frame):
        display_frame = frame.copy()
        roi_y1 = int(IMG_HEIGHT * 0.5)
        cv2.rectangle(display_frame, (0, roi_y1), (IMG_WIDTH, IMG_HEIGHT), (0,255,255), 2)
        with self.thread_data.lock:
            target_digit = self.thread_data.target_digit
            current_offset = self.thread_data.current_offset
            current_pid_output = self.thread_data.current_pid_output
            crossroad_detected = self.thread_data.crossroad_detected
            crossroad_decision = self.thread_data.crossroad_decision
            waiting_for_red_line = self.thread_data.waiting_for_red_line
            uart_status = self.thread_data.uart_status
            is_returning = self.thread_data.is_returning
            pause_control = self.thread_data.pause_control
        if target_digit is not None:
            cv2.putText(display_frame, f"Target: {target_digit}", (IMG_WIDTH-80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(display_frame, "Multi-Thread Mode", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if pause_control:
            cv2.putText(display_frame, "PAUSED by before_back", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif waiting_for_red_line:
            cv2.putText(display_frame, "WAITING FOR RED LINE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        elif crossroad_detected:
            cv2.putText(display_frame, f"CROSSROAD: {crossroad_decision or 'DECIDING'}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        elif current_offset is not None:
            cv2.putText(display_frame, "PID Line Following", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(display_frame, f"Offset: {current_offset}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if current_pid_output is not None:
                cv2.putText(display_frame, f"PID: {current_pid_output:.1f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cx = current_offset + IMG_WIDTH // 2
                cv2.circle(display_frame, (cx, roi_y1 + 20), 8, (0,255,0), -1)
                pid_indicator_x = int(IMG_WIDTH // 2 + current_pid_output)
                if 0 <= pid_indicator_x < IMG_WIDTH:
                    cv2.circle(display_frame, (pid_indicator_x, roi_y1 + 50), 6, (255,0,255), -1)
        else:
            cv2.putText(display_frame, "No red line detected", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        status_text = f"Q:{uart_status['queue_size']} ACK:{'Y' if uart_status['waiting_for_ack'] else 'N'}"
        cv2.putText(display_frame, status_text, (IMG_WIDTH-60, IMG_HEIGHT-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        if is_returning:
            cv2.putText(display_frame, "RETURN MODE", (10, IMG_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return display_frame

def main():
    import os
    if not os.path.exists(SERIAL_PORT):
        print(f"错误: GPIO串口设备 {SERIAL_PORT} 不存在")
        return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    print("正在初始化摄像头...")
    for i in range(5):
        ret, _ = cap.read()
        if ret:
            break
    if not ret:
        print("摄像头初始化失败")
        return
    try:
        uart = UartSender(port=SERIAL_PORT, baudrate=BAUDRATE)
        print(f"✓ GPIO串口连接成功: {SERIAL_PORT}")
    except Exception as e:
        print(f"✗ GPIO串口连接失败: {e}")
        cap.release()
        return
    print("等待下位机启动指令（weight_right）...")
    while True:
        resp = uart.read_response()
        if resp is not None and "weight_right" in resp:
            print("收到启动指令：weight_right，开始数字识别")
            break
    digit_detector = DigitDetector(IMG_WIDTH, IMG_HEIGHT, IMG_SIZE, DEVICE)
    digit_debugger = DigitDebugger(IMG_WIDTH, IMG_HEIGHT)
    line_follower = LineFollower(IMG_WIDTH, IMG_HEIGHT)
    print("\n选择PID参数预设:")
    list_presets()
    preset_choice = input("请输入预设名称 (直接回车使用balanced): ").strip()
    if not preset_choice:
        preset_choice = "balanced"
    preset_config = get_preset(preset_choice)
    print(f"✓ 使用PID预设: {preset_choice} - {preset_config['description']}")
    pid_controller = PIDController(
        kp=preset_config['kp'], 
        ki=preset_config['ki'], 
        kd=preset_config['kd'], 
        output_limits=(-PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT)
    )
    print(f"✓ PID控制器初始化完成 (Kp={preset_config['kp']}, Ki={preset_config['ki']}, Kd={preset_config['kd']})")
    print("\n选择模式:")
    print("1. 多线程运行模式 (推荐)")
    print("2. 数字识别调试模式")
    print("3. PID参数调试模式")
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    model = load_digit_model(MODEL_PATH, device=DEVICE, img_size=IMG_SIZE)
    if choice == "2":
        digit_debugger.debug_digit_recognition(cap, model, DEVICE, IMG_SIZE)
        cap.release()
        cv2.destroyAllWindows()
        uart.close()
        return
    elif choice == "3":
        _run_pid_debug_mode(cap, line_follower, pid_controller, preset_config, uart)
        return
    print("启动多线程模式...")
    thread_data = ThreadData()
    print("\n选择目标数字识别方法:")
    target_digit = digit_detector.recognize_target_digit(cap, model)
    if target_digit is not None:
        print(f"识别到目标数字为: {target_digit}")
        thread_data.target_digit = target_digit
    else:
        print("未能识别目标数字，请重启或调整数字位置")
        cap.release()
        uart.close()
        return
    camera_thread = CameraThread(thread_data, cap, save_dir='crossroad_images')
    vision_thread = VisionControlThread(thread_data, model, digit_detector, digit_debugger, 
                                        line_follower, pid_controller, camera_thread)
    comm_thread = CommunicationThread(thread_data, uart)
    display_thread = DisplayThread(thread_data)
    try:
        camera_thread.start()
        time.sleep(0.01)
        vision_thread.start()
        comm_thread.start()
        display_thread.start()
        display_thread.join()
        print("正在停止所有线程...")
        with thread_data.lock:
            thread_data.program_running = False
        camera_thread.join(timeout=2.0)
        vision_thread.join(timeout=2.0)
        comm_thread.join(timeout=2.0)
        print("✓ 所有线程已停止")
    except KeyboardInterrupt:
        print("程序被用户中断")
        with thread_data.lock:
            thread_data.program_running = False
    finally:
        print("正在关闭系统...")
        if uart.wait_for_queue_empty(timeout=3.0):
            print("✓ 所有待发送指令已完成")
        else:
            print("⚠️ 部分指令可能未完成发送")
        cap.release()
        uart.close()
        cv2.destroyAllWindows()
        print("✓ 系统已安全关闭")

def _run_pid_debug_mode(cap, line_follower, pid_controller, preset_config, uart):
    print("PID参数调试模式")
    print("使用以下键控制:")
    print("q/w: 调整Kp (减少/增加)")
    print("a/s: 调整Ki (减少/增加)")
    print("z/x: 调整Kd (减少/增加)")
    print("r: 重置PID控制器")
    print("ESC: 退出")
    current_kp = preset_config['kp']
    current_ki = preset_config['ki']
    current_kd = preset_config['kd']
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            display_frame = frame.copy()
            roi_y1 = int(IMG_HEIGHT * 0.5)
            cv2.rectangle(display_frame, (0, roi_y1), (IMG_WIDTH, IMG_HEIGHT), (0,255,255), 2)
            offset, found = line_follower.get_offset(frame)
            if found:
                pid_output = pid_controller.compute(0, offset)
                cv2.putText(display_frame, f"Offset: {offset}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(display_frame, f"PID Out: {pid_output:.1f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(display_frame, f"Kp:{current_kp:.2f} Ki:{current_ki:.3f} Kd:{current_kd:.2f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cx = offset + IMG_WIDTH // 2
                cv2.circle(display_frame, (cx, roi_y1 + 20), 5, (0,255,0), -1)
                pid_x = int(IMG_WIDTH // 2 + pid_output)
                if 0 <= pid_x < IMG_WIDTH:
                    cv2.circle(display_frame, (pid_x, roi_y1 + 40), 5, (0,255,255), -1)
            cv2.imshow('PID Debug', display_frame)
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                break
            elif key == ord('q'):
                current_kp = max(0.1, current_kp - 0.1)
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Kp调整为: {current_kp:.2f}")
            elif key == ord('w'):
                current_kp += 0.1
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Kp调整为: {current_kp:.2f}")
            elif key == ord('a'):
                current_ki = max(0.0, current_ki - 0.01)
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Ki调整为: {current_ki:.3f}")
            elif key == ord('s'):
                current_ki += 0.01
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Ki调整为: {current_ki:.3f}")
            elif key == ord('z'):
                current_kd = max(0.0, current_kd - 0.05)
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Kd调整为: {current_kd:.3f}")
            elif key == ord('x'):
                current_kd += 0.05
                pid_controller.set_tunings(current_kp, current_ki, current_kd)
                print(f"Kd调整为: {current_kd:.3f}")
            elif key == ord('r'):
                pid_controller.reset()
                print("PID控制器已重置")
    finally:
        cap.release()
        uart.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()