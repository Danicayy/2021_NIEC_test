#!/usr/bin/env python3
"""
测试串口通信和下位机回传数据的脚本
"""

import time
from uart_sender import UartSender

def test_uart_communication():
    """测试串口通信功能"""
    print("开始测试串口通信...")
    
    # 初始化串口 (根据你的实际串口设置)
    SERIAL_PORT = '/dev/ttyUSB0'  # Linux/树莓派
    # SERIAL_PORT = 'COM3'        # Windows
    BAUDRATE = 115200
    
    try:
        uart = UartSender(port=SERIAL_PORT, baudrate=BAUDRATE)
        print(f"串口 {SERIAL_PORT} 已连接")
        
        # 测试1: 发送左转指令并等待回传
        print("\n测试1: 发送左转指令...")
        response = uart.send_turn_and_wait_response('LEFT', timeout=3.0)
        if response:
            print(f"✓ 收到回传数据: {response}")
        else:
            print("✗ 未收到回传数据或超时")
        
        time.sleep(1)
        
        # 测试2: 发送右转指令并等待回传
        print("\n测试2: 发送右转指令...")
        response = uart.send_turn_and_wait_response('RIGHT', timeout=3.0)
        if response:
            print(f"✓ 收到回传数据: {response}")
        else:
            print("✗ 未收到回传数据或超时")
        
        time.sleep(1)
        
        # 测试3: 发送偏移量并监听回传
        print("\n测试3: 发送偏移量并监听回传...")
        uart.send_offset(50)
        time.sleep(0.1)
        response = uart.read_response()
        if response:
            print(f"✓ 收到回传数据: {response}")
        else:
            print("✗ 未收到回传数据")
        
        # 测试4: 持续监听10秒钟
        print("\n测试4: 持续监听下位机回传数据 (10秒)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            response = uart.read_response()
            if response:
                print(f"收到: {response}")
            time.sleep(0.1)
        
        uart.close()
        print("\n测试完成，串口已关闭")
        
    except Exception as e:
        print(f"串口测试出错: {e}")
        print("请检查:")
        print("1. 串口设备是否正确连接")
        print("2. 串口设备路径是否正确")
        print("3. 波特率是否匹配")
        print("4. 是否有足够的权限访问串口")

def monitor_uart_continuous():
    """持续监听下位机数据"""
    print("持续监听模式 (按Ctrl+C退出)...")
    
    SERIAL_PORT = '/dev/ttyUSB0'  # 根据实际情况修改
    BAUDRATE = 115200
    
    try:
        uart = UartSender(port=SERIAL_PORT, baudrate=BAUDRATE)
        print(f"开始监听串口 {SERIAL_PORT}")
        print("发送指令菜单:")
        print("l - 发送左转指令")
        print("r - 发送右转指令") 
        print("s - 发送搜索指令")
        print("q - 退出")
        
        while True:
            # 检查用户输入
            import select
            import sys
            
            # 检查是否有用户输入
            if select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'l':
                    print("发送左转指令...")
                    uart.send_turn('LEFT')
                elif user_input == 'r':
                    print("发送右转指令...")
                    uart.send_turn('RIGHT')
                elif user_input == 's':
                    print("发送搜索指令...")
                    uart.send_cmd('SEARCH')
                elif user_input == 'q':
                    break
            
            # 检查串口回传数据
            response = uart.read_response()
            if response:
                print(f"[{time.strftime('%H:%M:%S')}] 下位机回传: {response}")
            
            time.sleep(0.01)  # 短暂休眠
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"监听出错: {e}")
    finally:
        try:
            uart.close()
            print("串口已关闭")
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        monitor_uart_continuous()
    else:
        test_uart_communication()
