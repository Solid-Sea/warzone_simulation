# 战区模拟控制器
import time
import requests
import numpy as np
import threading
import json

# API服务器地址
API_URL = "http://localhost:5000/api"

class SimulationController:
    def __init__(self):
        """初始化模拟控制器"""
        self.running = False
        self.step = 0
        self.event_queue = []
        self.lock = threading.Lock()
        self.map_width = 80  # 地图宽度
        self.map_height = 60  # 地图高度
        self.shared_vision = {
            "red": np.zeros((self.map_height, self.map_width)),  # 使用实际地图尺寸
            "blue": np.zeros((self.map_height, self.map_width))
        }
        
    def init_simulation(self):
        """初始化模拟环境"""
        try:
            print(f"尝试连接到API服务器: {API_URL}/init")
            response = requests.post(f"{API_URL}/init", timeout=5.0)
            print(f"收到响应状态码: {response.status_code}")
            response.raise_for_status()
            print("模拟初始化成功，响应内容:", response.text)
            return True
        except requests.exceptions.RequestException as e:
            print(f"模拟初始化失败: {str(e)}")
            if hasattr(e, 'response') and e.response:
                print("错误响应内容:", e.response.text)
            return False
    
    def create_explosion(self, x, y, radius=4):
        """创建爆炸事件"""
        with self.lock:
            self.event_queue.append({
                "type": "explosion",
                "x": x,
                "y": y,
                "radius": radius
            })
    
    def move_unit(self, unit_id, x, y):
        """移动单位"""
        with self.lock:
            self.event_queue.append({
                "type": "move",
                "unit_id": unit_id,
                "x": x,
                "y": y
            })
    
    def attack(self, attacker_id, target_id):
        """单位攻击"""
        with self.lock:
            self.event_queue.append({
                "type": "attack",
                "attacker_id": attacker_id,
                "target_id": target_id
            })
    
    def get_state(self):
        """获取当前状态"""
        try:
            response = requests.get(f"{API_URL}/state", timeout=3.0)
            response.raise_for_status()
            
            # 添加原始响应内容检查
            print(f"原始响应内容: {response.text[:200]}...")  # 打印前200字符
            return response.json()
        except Exception as e:
            print(f"获取状态失败: {str(e)}")
            # 返回带默认值的结构避免崩溃
            return {
                "terrain": [[0]*80 for _ in range(60)],
                "units": [],
                "structures": []
            }
    
    def process_events(self):
        """处理事件队列"""
        with self.lock:
            if not self.event_queue:
                return
            
            # 处理事件
            for event in self.event_queue:
                try:
                    if event["type"] == "explosion":
                        response = requests.post(f"{API_URL}/explosion", json={
                            "x": event["x"],
                            "y": event["y"],
                            "radius": event["radius"]
                        })
                        if response.status_code == 200:
                            print(f"爆炸事件处理成功: ({event['x']}, {event['y']})")
                    
                    elif event["type"] == "move":
                        response = requests.post(f"{API_URL}/move_unit", json={
                            "unit_id": event["unit_id"],
                            "x": event["x"],
                            "y": event["y"]
                        })
                        if response.status_code == 200:
                            print(f"单位移动成功: 单位 {event['unit_id']} 到 ({event['x']}, {event['y']})")
                    
                    elif event["type"] == "attack":
                        response = requests.post(f"{API_URL}/attack", json={
                            "attacker_id": event["attacker_id"],
                            "target_id": event["target_id"]
                        })
                        if response.status_code == 200:
                            data = response.json()
                            if data["success"]:
                                print(f"攻击成功: 单位 {event['attacker_id']} 攻击了单位 {event['target_id']}")
                except Exception as e:
                    print(f"处理事件失败: {e}")
            
            # 清空队列
            self.event_queue = []
    
    def auto_simulation_step(self):
        """自动模拟步骤"""
        self.step += 1
        print(f"\n模拟步数: {self.step}")
        
        # 随机触发事件
        if self.step % 5 == 0:
            # 随机位置发生爆炸
            x = np.random.randint(10, 70)
            y = np.random.randint(10, 50)
            self.create_explosion(x, y, radius=4)
        
        # 随机单位移动
        if self.step % 3 == 0:
            # 获取当前状态
            state = self.get_state()
            if state:
                active_units = [u for u in state["units"] if u["status"] == "active"]
                if active_units:
                    unit = np.random.choice(active_units)
                    dx = np.random.randint(-3, 4)
                    dy = np.random.randint(-3, 4)
                    new_x = max(0, min(79, unit["position"][0] + dx))
                    new_y = max(0, min(59, unit["position"][1] + dy))
                    self.move_unit(unit["id"], new_x, new_y)
        
        # 随机攻击事件
        if self.step % 4 == 0:
            state = self.get_state()
            if state:
                active_units = [u for u in state["units"] if u["status"] == "active"]
                if len(active_units) >= 2:
                    attacker = np.random.choice(active_units)
                    # 找到最近的敌人
                    enemies = [u for u in active_units if u["id"] != attacker["id"]]
                    if enemies:
                        target = min(enemies, key=lambda u: 
                                    abs(u["position"][0] - attacker["position"][0]) + 
                                    abs(u["position"][1] - attacker["position"][1]))
                        self.attack(attacker["id"], target["id"])
    
    def run(self, max_steps=100):
        """运行模拟控制器"""
        if not self.init_simulation():
            return
        
        self.running = True
        step_count = 0
        
        while self.running and step_count < max_steps:
            # 自动生成事件
            self.auto_simulation_step()
            
            # 处理事件队列
            self.process_events()
            
            # 检查是否结束
            state = self.get_state()
            if state:
                active_units = [u for u in state["units"] if u["status"] == "active"]
                if len(active_units) < 2:
                    print("模拟结束: 一方部队已被歼灭")
                    self.running = False
            
            # 等待
            time.sleep(0.5)
            step_count += 1
        
        print("模拟控制器完成")

if __name__ == "__main__":
    controller = SimulationController()
    controller.run()
