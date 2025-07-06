from flask import Flask, jsonify

app = Flask(__name__)

# 初始化模拟数据
sim_state = {
    "terrain": [[0]*80 for _ in range(60)],
    "units": [
        {"id": 1, "position": [10,10], "camp": "red", "health": 100, "status": "active", "type": "infantry"},
        {"id": 2, "position": [60,50], "camp": "blue", "health": 100, "status": "active", "type": "infantry"}
    ],
    "structures": []
}

@app.route('/api/init', methods=['POST'])
def init_simulation():
    """初始化模拟环境"""
    # 重置单位状态
    for unit in sim_state["units"]:
        unit["health"] = 100
        unit["status"] = "active"
    return jsonify({"status": "success"}), 200

@app.route('/api/state')
def get_state():
    """获取当前状态"""
    return jsonify(sim_state)

@app.route('/api/move_unit', methods=['POST'])
def move_unit():
    """处理单位移动"""
    return jsonify({"status": "success"}), 200

@app.route('/api/attack', methods=['POST'])
def handle_attack():
    """处理攻击事件"""
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
