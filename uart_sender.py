import serial
import time
import threading
from queue import Queue

class UartSender:
    def __init__(self, port, baudrate):
        # GPIO串口配置
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.01,
            parity=serial.PARITY_NONE,  # 无校验位
            stopbits=serial.STOPBITS_ONE,  # 1个停止位
            bytesize=serial.EIGHTBITS,  # 8位数据位
            xonxoff=False,  # 关闭软件流控
            rtscts=False,   # 关闭硬件流控
            dsrdtr=False    # 关闭DSR/DTR流控
        )
        
        # 清空输入输出缓冲区
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        # 握手通信参数
        self.comm_frequency = 0.001  # 通信频率0.001秒 (1毫秒)
        self.handshake_timeout = 0.5  # 握手超时时间100毫秒
        self.waiting_for_ack = False  # 是否正在等待确认
        self.last_send_time = 0  # 上次发送时间
        self.command_queue = Queue()  # 命令队列
        self.running = True
        
        # 启动握手通信线程
        self.comm_thread = threading.Thread(target=self._communication_thread, daemon=True)
        self.comm_thread.start()
        
        print(f"GPIO串口已初始化: {port} @ {baudrate} baud (握手通信模式)")

    def _communication_thread(self):
        """握手通信线程"""
        current_command = None
        send_time = 0
        
        while self.running:
            current_time = time.time()

            
            # 检查是否有新命令需要发送
            if current_command is None and not self.command_queue.empty():
                current_command = self.command_queue.get()
                send_time = 0  # 重置发送时间，立即发送
            
            # 如果有命令且满足通信频率要求
            if current_command is not None and (current_time - send_time) >= self.comm_frequency:
                if not self.waiting_for_ack:
                    # 发送命令
                    self._send_raw_data(current_command['data'])
                    self.waiting_for_ack = True
                    send_time = current_time
                    #print(f"   发送握手数据: {current_command['description']}")
                    #print(f"   发送握手数据: {current_command['description']}")
                    print(f"   发送握手数据: {current_command['description']}")
                else:
                    # 检查是否收到确认
                    response = self._read_immediate_response()

                    if response and "ok" in response.lower():

                        #print(f"   收到确认: {response}")
                        #print(f"   收到确认: {response}")
                        print(f"   收到确认: {response}")
                        self.waiting_for_ack = False
                        current_command = None  # 命令发送完成
                    elif (current_time - send_time) > self.handshake_timeout:
                        # 超时重发
                        #print(f"   握手超时，重发: {current_command['description']}")
                        #print(f"   握手超时，重发: {current_command['description']}")
                        print(f"   握手超时，重发: {current_command['description']}")
                        self.waiting_for_ack = False
                        send_time = 0  # 重置发送时间
            
            time.sleep(0.0001)  # 短暂休眠避免CPU占用过高

    def _send_raw_data(self, data):
        """直接发送原始数据"""
        try:
            self.ser.write(data)
            self.ser.flush()
        except Exception as e:
            print(f"发送数据错误: {e}")

    def _read_immediate_response(self):
        """立即读取响应"""
        try:

            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                return data.decode('utf-8', errors='ignore').strip()
            return None
        except Exception as e:
            print(f"读取串口数据错误: {e}")
            return None

    def _add_command_to_queue(self, data, description):
        """将命令添加到队列"""
        command = {
            'data': data,
            'description': description
        }
        self.command_queue.put(command)

    def send_offset(self, offset):
        # 构造6字节十六进制数据包
        if offset is not None:
            original_offset = offset  # 保存原始offset用于调试
            head = 1 if offset >= 0 else 0
            abs_offset = abs(offset)
            
            # 添加offset范围检查
            if abs_offset > 80:  # 更新为新分辨率的限制 (原160*1/2=80)
                print(f"警告：offset绝对值 {abs_offset} 超过80，限制为80")
                abs_offset = 80
            
            # 构造6字节: [符号位, 0x00, 0x00, 0x00, 十位数字, 个位数字]
            # 例如偏移量40: [0x01, 0x00, 0x00, 0x00, 0x04, 0x00]
            tens_digit = abs_offset // 10
            ones_digit = abs_offset % 10
            hex_data = bytes([head, 0x00, 0x00, 0x00, tens_digit, ones_digit])
            
            # 调试输出
            description = f"Offset数据包 - 原始:{original_offset}, 处理后:{offset if abs_offset <= 80 else (80 if offset > 0 else -80)}, 符号:{head}, 偏移:{abs_offset}"
            print(f"   准备发送Offset: {description}")
            print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
        else:
            # 红线丢失信号: 0x666666
            hex_data = bytes([0x66, 0x66, 0x66, 0x66, 0x66, 0x66])
            description = "红线丢失信号"
            print(f"   准备发送红线丢失: {description}")
            print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
        
        # 添加到命令队列
        self._add_command_to_queue(hex_data, description)

    def send_pid_control(self, pid_output):
        """
        发送PID控制值
        
        Args:
            pid_output: PID控制器输出值 (-100 到 100)
        """
        if pid_output is not None:
            # 将PID输出值限制在[-100, 100]范围内
            pid_output = max(-100, min(100, pid_output))
            
            # 符号位: 1表示正值(右转), 0表示负值(左转)
            head = 1 if pid_output >= 0 else 0
            abs_output = abs(pid_output)
            
            # 构造6字节: [符号位, 0x01, 0x00, 0x00, 十位数字, 个位数字]
            # 使用0x01作为PID控制标识符
            tens_digit = int(abs_output) // 10
            ones_digit = int(abs_output) % 10
            hex_data = bytes([head, 0x01, 0x00, 0x00, tens_digit, ones_digit])
            
            # 调试输出
            description = f"PID控制包 - 输出:{pid_output:.2f}, 符号:{head}, 绝对值:{abs_output:.2f}"
            # print(f"   准备发送PID控制: {description}")
            print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
        else:
            # 无效PID输出，发送停止信号
            hex_data = bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x00])
            description = "PID无效输出，发送停止"
            # print(f"   准备发送PID停止: {description}")
            print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
        
        # 添加到命令队列
        self._add_command_to_queue(hex_data, description)

    def send_cmd(self, cmd):
        # 将指令转换为6字节十六进制格式
        if isinstance(cmd, (int, float)):
            original_cmd = cmd  # 保存原始指令用于调试
            cmd_int = int(cmd)
            # 将数字转换为6位字符串，不足6位前面补0
            cmd_str = f"{cmd_int:06d}"
            # 每一位数字直接转换为对应的字节值
            hex_data = bytes([int(digit) for digit in cmd_str])
            
            # 调试输出
            description = f"指令数据包 - 原始:{original_cmd}, 6位数字:{cmd_str}"
            #print(f"   准备发送指令: {description}")
            if isinstance(original_cmd, int) and original_cmd > 0:
                pass
                #print(f"   十六进制表示: 0x{original_cmd:08X}")
            #print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
            
            # 添加到命令队列
            self._add_command_to_queue(hex_data, description)
        else:
            # 特殊字符串指令（如SEARCH）保持原样发送
            string_data = str(cmd).encode("utf-8")
            
            # 调试输出
            description = f"字符串指令 - '{cmd}'"
            print(f"   准备发送字符串: {description}")
            print(f"   UTF-8编码: {list(string_data)}, 十六进制: {string_data.hex().upper()}")
            
            # 添加到命令队列
            self._add_command_to_queue(string_data, description)

    # def send_turn(self, cmd):
    #     if cmd == 'LEFT':
    #         # 左转指令: 0x200002 (十六进制, 6字节)
    #         hex_data = bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x02])
            
    #         # 调试输出
    #         description = f"转弯指令 - {cmd} (左转), 代码:0x02000002"
    #         print(f"   准备发送转弯: {description}")
    #         print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
            
    #         # 添加到命令队列
    #         self._add_command_to_queue(hex_data, description)
    #     elif cmd == 'RIGHT':
    #         # 右转指令: 0x200001 (十六进制, 6字节)
    #         hex_data = bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x01])
            
    #         # 调试输出
    #         description = f"转弯指令 - {cmd} (右转), 代码:0x02000001"
    #         print(f"   准备发送转弯: {description}")
    #         print(f"   字节数组: {list(hex_data)}, 十六进制: {hex_data.hex().upper()}")
            
    #         # 添加到命令队列
    #         self._add_command_to_queue(hex_data, description)
    #     else:
    #         print(f"  未知转弯指令: {cmd}")

    def read_response(self):
        """读取下位机回传的数据（兼容旧接口）"""
        return self._read_immediate_response()

    def read_line_response(self):
        """按行读取下位机回传的数据"""
        try:
            if self.ser.in_waiting > 0:
                # 按行读取数据
                data = self.ser.readline()
                return data.decode('utf-8', errors='ignore').strip()
            return None
        except Exception as e:
            print(f"读取串口数据错误: {e}")
            return None

    def send_turn_and_wait_response(self, cmd, timeout=0.2):
        """发送转弯指令并等待回传数据（已废弃，使用握手通信）"""
        print(f"⚠️ send_turn_and_wait_response已废弃，请使用send_turn，握手通信会自动处理确认")
        self.send_turn(cmd)
        return "ok"  # 模拟返回确认

    def get_queue_status(self):
        """获取命令队列状态"""
        return {
            'queue_size': self.command_queue.qsize(),
            'waiting_for_ack': self.waiting_for_ack,
            'last_send_time': self.last_send_time
        }

    def clear_queue(self):
        """清空命令队列"""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        print("命令队列已清空")

    def wait_for_queue_empty(self, timeout=5.0):
        """等待命令队列清空"""
        start_time = time.time()
        while not self.command_queue.empty() or self.waiting_for_ack:
            if time.time() - start_time > timeout:
                print(f"⚠️ 等待队列清空超时 ({timeout}秒)")
                return False
            time.sleep(0.01)
        return True

    def close(self):
        """关闭串口连接"""
        self.running = False
        if hasattr(self, 'comm_thread') and self.comm_thread.is_alive():
            self.comm_thread.join(timeout=1.0)
        self.ser.close()
        print("GPIO串口已关闭，握手通信线程已停止")