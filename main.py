# 战区模拟器主入口
import random
import time
import pygame
import numpy as np
from core.map_generator import MapGenerator
from core.terrain_engine import TerrainEngine
from entities.unit import Unit
from entities.structure import Structure
from visualization.renderer import Renderer

class BattleSimulator:
    """优化的战场模拟器"""
    
    def __init__(self, width=80, height=60):
        self.width = width
        self.height = height
        self.map_gen = MapGenerator(width=width, height=height)
        self.terrain_engine = TerrainEngine(self.map_gen.generate_terrain("desert"))
        self.units = []
        self.structures = []
        self.renderer = None
        self.step = 0
        
    def deploy_units(self, camp, start_x, start_y, counts):
        """统一单位部署"""
        unit_types = ["infantry"]*counts[0] + ["tank"]*counts[1] + ["artillery"]*counts[2] + ["engineer"]*counts[3]
        for i, utype in enumerate(unit_types):
            x = start_x + i % 5
            y = start_y + i // 5
            unit = Unit(len(self.units)+1, utype, position=(x, y), camp=camp)
            self.units.append(unit)
    
    def initialize_battle(self):
        """初始化战场"""
        print("生成战场地图...")
        print("部署军事单位...")
        
        # 部署双方部队
        self.deploy_units("red", 10, 10, [15, 3, 2, 3])
        self.deploy_units("blue", 60, 50, [15, 3, 2, 3])
        
        # 创建关键建筑
        self.structures = [
            Structure(1, "house", position=(30, 20), dimensions=(3, 2)),
            Structure(2, "bunker", position=(40, 35), dimensions=(2, 2)),
            Structure(3, "bridge", position=(25, 40), dimensions=(5, 1))
        ]
        
        # 初始化渲染器
        self.renderer = Renderer(width=self.width, height=self.height, cell_size=12)
        pygame.font.init()
        try:
            font = pygame.font.Font("simhei.ttf", 16)
        except:
            font = pygame.font.SysFont("SimHei", 16)
        self.renderer.font = font
        self.renderer.small_font = pygame.font.SysFont("SimHei", 12)
    
    def get_active_units(self, camp=None):
        """获取活跃单位"""
        units = [u for u in self.units if u.is_active()]
        if camp:
            units = [u for u in units if u.camp == camp]
        return units
    
    def check_victory(self):
        """检查胜利条件"""
        red_units = self.get_active_units("red")
        blue_units = self.get_active_units("blue")
        central_buildings = [s for s in self.structures if 25 <= s.position[0] <= 45 and 15 <= s.position[1] <= 35]
        
        # 全歼胜利
        if not red_units and blue_units:
            return "蓝方胜利 - 全歼红方部队"
        if not blue_units and red_units:
            return "红方胜利 - 全歼蓝方部队"
        
        # 建筑控制胜利
        if central_buildings:
            red_control = sum(1 for b in central_buildings 
                            if sum(1 for u in red_units 
                                 if abs(u.position[0]-b.position[0]) <= 3 and abs(u.position[1]-b.position[1]) <= 3) > 0)
            blue_control = sum(1 for b in central_buildings 
                             if sum(1 for u in blue_units 
                                  if abs(u.position[0]-b.position[0]) <= 3 and abs(u.position[1]-b.position[1]) <= 3) > 0)
            
            if red_control >= len(central_buildings) * 0.7:
                return "红方胜利 - 占领关键建筑"
            if self.step > 200 and blue_control >= len(central_buildings) * 0.9:
                return "蓝方胜利 - 成功坚守阵地"
        
        return None
    
    def process_artillery_attacks(self):
        """处理炮兵攻击"""
        artilleries = [u for u in self.units if u.type == "artillery" and u.is_active()]
        for artillery in artilleries:
            enemies = [u for u in self.units if u.camp != artillery.camp and u.is_active()]
            if enemies:
                target = random.choice(enemies)
                attack_data = artillery.explosive_attack(target.position)
                if attack_data:
                    self.terrain_engine.add_crater(*attack_data["position"], radius=attack_data["radius"])
                    self._apply_explosion_damage(attack_data, artillery)
    
    def _apply_explosion_damage(self, attack_data, artillery):
        """应用爆炸伤害"""
        for unit in self.units:
            if unit.is_active():
                dx = abs(unit.position[0] - attack_data["position"][0])
                dy = abs(unit.position[1] - attack_data["position"][1])
                distance = dx + dy
                
                if distance <= attack_data["radius"] * 2:
                    damage = int(artillery.damage * attack_data["damage_factor"] * 
                               max(0, 1 - distance/(attack_data["radius"]*2)))
                    unit.take_damage(damage)
    
    def process_unit_movement(self):
        """处理单位移动"""
        for unit in self.units:
            if unit.is_active():
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                new_x = max(0, min(self.width-1, unit.position[0] + dx))
                new_y = max(0, min(self.height-1, unit.position[1] + dy))
                unit.move_to((new_x, new_y))
    
    def process_combat(self):
        """处理战斗"""
        active_units = self.get_active_units()
        for attacker in active_units:
            if np.random.random() < 0.5:
                enemies = [u for u in active_units if u.camp != attacker.camp]
                if enemies:
                    enemies.sort(key=lambda u: abs(u.position[0]-attacker.position[0]) + abs(u.position[1]-attacker.position[1]))
                    target = enemies[0]
                    distance = abs(target.position[0]-attacker.position[0]) + abs(target.position[1]-attacker.position[1])
                    if distance <= attacker.attack_range:
                        attacker.attack(target)
    
    def run_simulation(self, max_steps=300):
        """运行完整模拟"""
        self.initialize_battle()
        print("开始作战模拟...")
        
        running = True
        while running and self.step < max_steps:
            self.step += 1
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            # 游戏逻辑
            if self.step % 6 == 0:
                self.process_artillery_attacks()
            if self.step % 3 == 0:
                self.process_unit_movement()
            if self.step % 4 == 0:
                self.process_combat()
            
            # 渲染
            current_terrain = self.terrain_engine.get_current_terrain()
            self.renderer.render_all(current_terrain, self.units, self.structures)
            pygame.time.delay(100)
            
            # 检查胜利
            victory = self.check_victory()
            if victory:
                print(f"模拟结束: {victory}!")
                running = False
        
        print("作战模拟完成!")

def main():
    simulator = BattleSimulator()
    simulator.run_simulation()

if __name__ == "__main__":
    main()
