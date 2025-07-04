# 战区模拟数据处理API服务器
from flask import Flask, request, jsonify
import numpy as np
from core.map_generator import MapGenerator
from core.terrain_engine import TerrainEngine
from core.pathfinding import Pathfinder
from entities.unit import Unit
from entities.structure import Structure

app = Flask(__name__)

# 全局状态
terrain_engine = None
units = []
structures = []
pathfinder = None

def init_simulation():
    """初始化模拟环境"""
    global terrain_engine, units, structures, pathfinder
    
    # 生成地图
    map_gen = MapGenerator(width=80, height=60)
    base_terrain = map_gen.generate_terrain("desert")
    terrain_engine = TerrainEngine(base_terrain)
    pathfinder = Pathfinder(terrain_engine.get_current_terrain())
    
    # 创建单位 - 每方20个单位（15步兵，3坦克，2火炮）
    units = []
    unit_id = 1
    
    # 红方单位
    for i in range(15):  # 步兵
        units.append(Unit(unit_id, "infantry", (np.random.randint(5, 35), np.random.randint(5, 25)), camp="red"))
        unit_id += 1
    for i in range(3):   # 坦克
        units.append(Unit(unit_id, "tank", (np.random.randint(5, 35), np.random.randint(5, 25)), camp="red"))
        unit_id += 1
    for i in range(2):   # 火炮
        units.append(Unit(unit_id, "artillery", (np.random.randint(5, 35), np.random.randint(5, 25)), camp="red"))
        unit_id += 1
        
    # 蓝方单位
    for i in range(15):  # 步兵
        units.append(Unit(unit_id, "infantry", (np.random.randint(45, 75), np.random.randint(35, 55)), camp="blue"))
        unit_id += 1
    for i in range(3):   # 坦克
        units.append(Unit(unit_id, "tank", (np.random.randint(45, 75), np.random.randint(35, 55)), camp="blue"))
        unit_id += 1
    for i in range(2):   # 火炮
        units.append(Unit(unit_id, "artillery", (np.random.randint(45, 75), np.random.randint(35, 55)), camp="blue"))
        unit_id += 1
    
    # 创建建筑
    structures = [
        Structure(1, "house", position=(30, 20), dimensions=(3, 2)),
        Structure(2, "bunker", position=(40, 35), dimensions=(2, 2)),
        Structure(3, "bridge", position=(25, 40), dimensions=(5, 1))
    ]

@app.route('/api/init', methods=['POST'])
def init_api():
    """初始化模拟"""
    init_simulation()
    return jsonify({"status": "initialized"})

@app.route('/api/explosion', methods=['POST'])
def create_explosion():
    """处理爆炸事件"""
    data = request.json
    x = data['x']
    y = data['y']
    radius = data.get('radius', 4)
    
    # 更新地形
    terrain_engine.add_crater(x, y, radius)
    
    # 计算伤害
    damaged_units = []
    for unit in units:
        if unit.is_active():
            distance = abs(unit.position[0] - x) + abs(unit.position[1] - y)
            if distance < 8:
                damage = max(30, 60 - distance * 5)
                unit.take_damage(damage)
                damaged_units.append({
                    "id": unit.id,
                    "damage": damage,
                    "position": unit.position,
                    "health": unit.health
                })
    
    return jsonify({
        "terrain": terrain_engine.get_current_terrain().tolist(),
        "damaged_units": damaged_units
    })

@app.route('/api/move_unit', methods=['POST'])
def move_unit():
    """移动单位"""
    data = request.json
    unit_id = data['unit_id']
    new_x = data['x']
    new_y = data['y']
    
    unit = next((u for u in units if u.id == unit_id), None)
    if unit and unit.is_active():
        unit.move_to((new_x, new_y))
        return jsonify({
            "id": unit.id,
            "position": unit.position,
            "status": "success"
        })
    return jsonify({"status": "failed"})

@app.route('/api/attack', methods=['POST'])
def attack():
    """处理攻击事件"""
    data = request.json
    attacker_id = data['attacker_id']
    target_id = data['target_id']
    
    attacker = next((u for u in units if u.id == attacker_id), None)
    target = next((u for u in units if u.id == target_id), None)
    
    if attacker and target and attacker.is_active():
        success = attacker.attack(target)
        return jsonify({
            "success": success,
            "attacker": attacker_id,
            "target": target_id,
            "target_health": target.health if target else 0
        })
    return jsonify({"success": False})

@app.route('/api/state', methods=['GET'])
def get_state():
    """获取当前状态"""
    return jsonify({
        "terrain": terrain_engine.get_current_terrain().tolist(),
        "units": [{
            "id": u.id,
            "type": u.type,
            "position": u.position,
            "health": u.health,
            "status": u.status,
            "camp": u.camp  # 添加阵营信息
        } for u in units],
        "structures": [{
            "id": s.id,
            "type": s.type,
            "position": s.position,
            "dimensions": s.dimensions,
            "health": s.health,
            "status": s.status
        } for s in structures]
    })

if __name__ == '__main__':
    init_simulation()
    app.run(port=5000)
