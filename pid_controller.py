import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100)):
        """
        PID控制器
        
        Args:
            kp: 比例系数 (Proportional gain)
            ki: 积分系数 (Integral gain) 
            kd: 微分系数 (Derivative gain)
            output_limits: 输出限制 (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.output_min, self.output_max = output_limits
        
        # 内部状态变量
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        self.first_run = True
        
    def compute(self, setpoint, measurement):
        """
        计算PID输出
        
        Args:
            setpoint: 目标值 (对于循迹，通常是0，表示偏差为0)
            measurement: 当前测量值 (当前的位置偏差)
            
        Returns:
            control_output: 控制输出值
        """
        current_time = time.time()
        
        # 计算误差
        error = setpoint - measurement
        
        if self.first_run:
            # 第一次运行，初始化
            self.prev_error = error
            self.prev_time = current_time
            self.first_run = False
            self.integral = 0.0
            dt = 0.02  # 假设初始dt为20ms
        else:
            # 计算时间差
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.02  # 防止除零，设置最小时间间隔
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / dt
        derivative_term = self.kd * derivative
        
        # PID输出
        output = proportional + integral_term + derivative_term
        
        # 输出限幅
        output = max(self.output_min, min(self.output_max, output))
        
        # 积分饱和处理 (Anti-windup)
        if output >= self.output_max or output <= self.output_min:
            self.integral -= error * dt  # 回退积分项，防止积分饱和
        
        # 更新历史值
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """重置PID控制器状态"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        self.first_run = True
    
    def set_tunings(self, kp, ki, kd):
        """动态调整PID参数"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def set_output_limits(self, min_output, max_output):
        """设置输出限制"""
        self.output_min = min_output
        self.output_max = max_output
