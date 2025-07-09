# 战区模拟控制器
import time
import numpy as np
import threading


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
        print("本地模拟环境初始化成功")
        return True
    
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
        """获取本地模拟状态"""
        # 简化状态返回
        return {
            "terrain": [[0]*80 for _ in range(60)],
            "units": [
                {"id": 1, "position": [10,10], "camp": "red", "health": 100, "status": "active", "type": "infantry"},
                {"id": 2, "position": [60,50], "camp": "blue", "health": 100, "status": "active", "type": "infantry"}
            ],
            "structures": []
        }
    
    def call_artillery(self, position):
        """模拟炮兵支援"""
        print(f"模拟炮兵支援在位置: {position}")
        self.create_explosion(position[0], position[1], radius=8)
    
    def process_events(self):
        """处理事件队列"""
        with self.lock:
            if not self.event_queue:
                return
            
            # 处理事件（减少控制台输出）
            for event in self.event_queue:
                try:
                    if event["type"] == "explosion":
                        pass  # 不再打印爆炸信息
                    elif event["type"] == "move":
                        pass  # 不再打印移动信息
                    elif event["type"] == "attack":
                        pass  # 不再打印攻击信息
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
