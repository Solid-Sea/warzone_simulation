# 战区模拟器主入口
import numpy as np
import time
import pygame
from core.map_generator import MapGenerator
from core.terrain_engine import TerrainEngine
from entities.unit import Unit
from entities.structure import Structure
from visualization.renderer import Renderer

def main():
    # 初始化地图
    print("生成战场地图...")
    map_gen = MapGenerator(width=80, height=60)
    base_terrain = map_gen.generate_terrain("desert")
    terrain_engine = TerrainEngine(base_terrain)
    
    # 创建军事单位（红方和蓝方）20v20 + 3工兵
    print("部署军事单位...")
    units = []
    unit_id = 1
    
    # 红方单位 (15步兵 + 3坦克 + 2炮兵 + 3工兵)
    for i in range(15):
        units.append(Unit(unit_id, "infantry", position=(10 + i % 5, 10 + i // 5), camp="red"))
        unit_id += 1
    for i in range(3):
        units.append(Unit(unit_id, "tank", position=(15 + i, 15), camp="red"))
        unit_id += 1
    for i in range(2):
        units.append(Unit(unit_id, "artillery", position=(20, 5 + i*2), camp="red"))
        unit_id += 1
    for i in range(3):
        units.append(Unit(unit_id, "engineer", position=(10, 20 + i), camp="red"))
        unit_id += 1
    
    # 蓝方单位 (15步兵 + 3坦克 + 2炮兵 + 3工兵)
    for i in range(15):
        units.append(Unit(unit_id, "infantry", position=(60 + i % 5, 50 + i // 5), camp="blue"))
        unit_id += 1
    for i in range(3):
        units.append(Unit(unit_id, "tank", position=(55 + i, 45), camp="blue"))
        unit_id += 1
    for i in range(2):
        units.append(Unit(unit_id, "artillery", position=(70, 40 + i*2), camp="blue"))
        unit_id += 1
    for i in range(3):
        units.append(Unit(unit_id, "engineer", position=(65, 55 + i), camp="blue"))
        unit_id += 1
    
    # 创建建筑
    print("建造战场结构...")
    structures = [
        Structure(1, "house", position=(30, 20), dimensions=(3, 2)),
        Structure(2, "bunker", position=(40, 35), dimensions=(2, 2)),
        Structure(3, "bridge", position=(25, 40), dimensions=(5, 1))
    ]
    
    # 初始化渲染器并设置中文字体
    renderer = Renderer(width=80, height=60, cell_size=12)
    pygame.font.init()
    # 确保使用支持中文的字体
    try:
        font = pygame.font.Font("simhei.ttf", 16)  # 尝试加载黑体
    except:
        font = pygame.font.SysFont("SimHei", 16)  # 回退到系统黑体
    renderer.font = font
    renderer.small_font = pygame.font.SysFont("SimHei", 12)
    
    # 模拟循环
    print("开始作战模拟...")
    simulation_running = True
    step = 0
    
    while simulation_running:
        step += 1
        print(f"\n模拟步数: {step}")
        
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    simulation_running = False
        
        # 炮兵攻击事件
        if step % 6 == 0 and simulation_running:
            # 获取所有活跃的炮兵单位
            artilleries = [u for u in units if u.type == "artillery" and u.is_active()]
            
            for artillery in artilleries:
                # 寻找攻击目标
                enemies = [u for u in units if u.camp != artillery.camp and u.is_active()]
                if enemies:
                    # 选择射程内的随机目标
                    target = np.random.choice(enemies)
                    attack_data = artillery.explosive_attack(target.position)
                    
                    if attack_data:
                        # 添加弹坑
                        terrain_engine.add_crater(*attack_data["position"], radius=attack_data["radius"])
                        
                        # 渲染爆炸特效
                        for i in range(3):
                            current_terrain = terrain_engine.get_current_terrain()
                            renderer.render_all(current_terrain, units, structures)
                            renderer.render_explosion(*attack_data["position"], 
                                                     radius=attack_data["radius"], 
                                                     intensity=1.0 - i*0.3)
                            pygame.display.flip()
                            pygame.time.delay(50)
                        
                        # 计算爆炸伤害
                        for unit in units:
                            if unit.is_active():
                                dx = abs(unit.position[0] - attack_data["position"][0])
                                dy = abs(unit.position[1] - attack_data["position"][1])
                                distance = dx + dy
                                
                                if distance <= attack_data["radius"] * 2:
                                    damage = int(artillery.damage * attack_data["damage_factor"] * (1 - distance/(attack_data["radius"]*2)))
                                    unit.take_damage(damage)
                                    
                                    # 记录击杀消息
                                    if unit.health <= 0:
                                        kill_msg = f"单位#{artillery.id}({artillery.camp}) 击毁 单位#{unit.id}({unit.camp})"
                                        renderer.add_info_message(kill_msg)
        
        # 随机单位移动
        if step % 3 == 0:
            for unit in units:
                if unit.is_active():
                    dx = np.random.randint(-3, 4)
                    dy = np.random.randint(-3, 4)
                    new_x = max(0, min(79, unit.position[0] + dx))
                    new_y = max(0, min(59, unit.position[1] + dy))
                    unit.move_to((new_x, new_y))
        
        # 阵营对抗攻击事件
        if step % 4 == 0:
            active_units = [u for u in units if u.is_active()]
            
            for attacker in active_units:
                # 只处理每4步中50%的单位
                if np.random.random() < 0.5:
                    # 找到最近的敌方单位
                    enemies = [
                        u for u in active_units 
                        if u.camp != attacker.camp and u.id != attacker.id
                    ]
                    
                    if enemies:
                        # 按距离排序并选择最近的
                        enemies.sort(key=lambda u: abs(u.position[0]-attacker.position[0]) + abs(u.position[1]-attacker.position[1]))
                        target = enemies[0]
                        
                        # 检查是否在攻击范围内
                        distance = abs(target.position[0]-attacker.position[0]) + abs(target.position[1]-attacker.position[1])
                        if distance <= attacker.attack_range:
                            if attacker.attack(target):
                                print(f"[{attacker.camp}] 单位 {attacker.id} ({attacker.type}) 攻击了 [{target.camp}] 单位 {target.id} ({target.type})!")
                        
                        # 显示攻击效果
                        start_pos = (int((attacker.position[0] + 0.5) * renderer.cell_size), 
                                    int((attacker.position[1] + 0.5) * renderer.cell_size))
                        end_pos = (int((target.position[0] + 0.5) * renderer.cell_size), 
                                  int((target.position[1] + 0.5) * renderer.cell_size))
                        
                        # 绘制攻击线
                        current_terrain = terrain_engine.get_current_terrain()
                        renderer.render_all(current_terrain, units, structures)
                        pygame.draw.line(renderer.screen, (255, 165, 0), start_pos, end_pos, 2)
                        pygame.display.flip()
                        time.sleep(0.1)
                        
                        # 显示伤害特效
                        damage_surface = pygame.Surface((renderer.cell_size*2, renderer.cell_size*2), pygame.SRCALPHA)
                        pygame.draw.circle(damage_surface, (255, 0, 0, 150), 
                                         (renderer.cell_size, renderer.cell_size), 
                                         int(renderer.cell_size * 0.8))
                        renderer.screen.blit(damage_surface, (end_pos[0]-renderer.cell_size, end_pos[1]-renderer.cell_size))
                        pygame.display.flip()
                        time.sleep(0.1)
        
        # 渲染当前状态
        if simulation_running:
            current_terrain = terrain_engine.get_current_terrain()
            renderer.render_all(current_terrain, units, structures)
            pygame.time.delay(100)  # 使用delay代替sleep，更可靠
        
        # 检查胜利条件
        red_units = [u for u in units if u.camp == "red" and u.is_active()]
        blue_units = [u for u in units if u.camp == "blue" and u.is_active()]
        central_buildings = [s for s in structures if 25 <= s.position[0] <= 45 and 15 <= s.position[1] <= 35]
        
        # 蓝方胜利条件：坚守建筑且红方兵力耗尽
        if len(red_units) == 0 and len(blue_units) > 0:
            print("模拟结束: 蓝方胜利 - 全歼红方部队!")
            simulation_running = False
            
        # 红方胜利条件：占领所有建筑或全歼蓝方
        elif len(blue_units) == 0:
            print("模拟结束: 红方胜利 - 全歼蓝方部队!")
            simulation_running = False
        else:
            # 检查建筑占领状态
            red_controlled = 0
            blue_controlled = 0
            for building in central_buildings:
                # 计算周围单位数量
                red_nearby = sum(1 for u in red_units 
                               if abs(u.position[0] - building.position[0]) <= 3 
                               and abs(u.position[1] - building.position[1]) <= 3)
                blue_nearby = sum(1 for u in blue_units 
                                if abs(u.position[0] - building.position[0]) <= 3 
                                and abs(u.position[1] - building.position[1]) <= 3)
                                
                if red_nearby > blue_nearby * 1.5:
                    red_controlled += 1
                elif blue_nearby > red_nearby * 1.5:
                    blue_controlled += 1
            
            # 胜利判定
            if red_controlled >= len(central_buildings) * 0.7:
                print("模拟结束: 红方胜利 - 占领关键建筑!")
                simulation_running = False
            elif step > 200 and blue_controlled >= len(central_buildings) * 0.9:
                print("模拟结束: 蓝方胜利 - 成功坚守阵地!")
                simulation_running = False
    
    print("作战模拟完成!")

if __name__ == "__main__":
    main()
