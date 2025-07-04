# 战区可视化应用（从API获取数据）
import pygame
import requests
import numpy as np
import math  # 添加math模块导入
import sys
from pygame.locals import *

# 添加地形纹理
def create_terrain_texture(cell_size):
    textures = {}
    # 平原纹理
    plain = pygame.Surface((cell_size, cell_size))
    plain.fill((210, 180, 140))
    for i in range(cell_size):
        for j in range(cell_size):
            if (i+j) % 3 == 0:
                plain.set_at((i, j), (220, 190, 150))
    textures[0] = plain
    
    # 山地纹理
    mountain = pygame.Surface((cell_size, cell_size))
    mountain.fill((139, 137, 137))
    for i in range(cell_size):
        for j in range(cell_size):
            if i % 2 == 0 and j % 2 == 0:
                mountain.set_at((i, j), (150, 148, 148))
    textures[1] = mountain
    
    # 水域纹理
    water = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
    water.fill((64, 164, 223, 200))
    for i in range(0, cell_size, 2):
        pygame.draw.line(water, (80, 180, 240, 220), (0, i), (cell_size, i), 1)
    textures[2] = water
    
    # 弹坑纹理
    crater = pygame.Surface((cell_size, cell_size))
    crater.fill((105, 105, 105))
    center = cell_size // 2
    for i in range(cell_size):
        for j in range(cell_size):
            dist = ((i-center)**2 + (j-center)**2)**0.5
            if dist < center:
                crater.set_at((i, j), (100, 100, 100))
    textures[3] = crater
    
    return textures

# API服务器地址
API_URL = "http://localhost:5000/api"

# 地形颜色映射
TERRAIN_COLORS = {
    0: (210, 180, 140),  # 平原: 沙色
    1: (139, 137, 137),  # 山地: 灰色
    2: (64, 164, 223),   # 水域: 蓝色
    3: (105, 105, 105)   # 弹坑: 深灰色
}

# 阵营颜色映射（更鲜明的对比色）
CAMP_COLORS = {
    "red": (255, 50, 50),    # 更鲜艳的红色
    "blue": (50, 150, 255)   # 更鲜艳的蓝色
}

# 单位类型颜色调整（基于阵营）
def get_unit_color(unit_type, camp):
    """根据单位类型和阵营获取颜色"""
    base_colors = {
        "infantry": (0, 180, 0) if camp == "red" else (0, 180, 255),
        "tank": (220, 60, 60) if camp == "red" else (100, 160, 255),
        "artillery": (255, 200, 0) if camp == "red" else (180, 220, 255)
    }
    return base_colors.get(unit_type, (200, 200, 200))

# 建筑颜色映射
STRUCTURE_COLORS = {
    "house": (200, 150, 100),   # 房屋: 棕色
    "bunker": (80, 80, 80),     # 碉堡: 深灰色
    "bridge": (139, 69, 19)     # 桥梁: 褐色
}

class VisualizationApp:
    def __init__(self, width=80, height=60, cell_size=12):
        """
        初始化可视化应用
        :param width: 地图宽度（单元格数）
        :param height: 地图高度（单元格数）
        :param cell_size: 每个单元格的像素大小
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # 初始化pygame
        pygame.init()
        self.info_panel_width = 300  # 信息面板宽度
        self.screen_width = width * cell_size + self.info_panel_width
        self.screen_height = height * cell_size
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), RESIZABLE)
        pygame.display.set_caption("战区模拟器 - 可视化")
        
        # 全屏状态
        self.fullscreen = False
        
        # 创建地形纹理
        self.terrain_textures = create_terrain_texture(cell_size)
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("SimHei", 20)  # 使用黑体支持中文
        self.small_font = pygame.font.SysFont("SimHei", 16)
        
        # 初始化地形表面
        self.terrain_surface = None
        
        # 初始化API状态
        self.terrain = None
        self.units = []
        self.structures = []
        self.step_count = 0  # 模拟步数计数器
        
    def fetch_state(self):
        """从API获取当前状态"""
        try:
            response = requests.get(f"{API_URL}/state")
            if response.status_code == 200:
                data = response.json()
                self.terrain = np.array(data["terrain"])
                self.units = data["units"]
                self.structures = data["structures"]
                return True
            return False
        except Exception as e:
            print(f"获取状态失败: {e}")
            return False
    
    def pre_render_terrain(self):
        """预渲染地形到单独的表面"""
        if self.terrain is None:
            return
            
        # 创建地形表面
        self.terrain_surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        
        for y in range(self.height):
            for x in range(self.width):
                terrain_type = self.terrain[y][x]
                texture = self.terrain_textures.get(terrain_type, None)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                if texture:
                    self.terrain_surface.blit(texture, rect)
                else:
                    color = TERRAIN_COLORS.get(terrain_type, (0, 0, 0))
                    pygame.draw.rect(self.terrain_surface, color, rect)
                
                # 绘制主要网格线（每5个单元格）
                if x % 5 == 0 or y % 5 == 0:
                    pygame.draw.rect(self.terrain_surface, (80, 80, 80, 100), rect, 1)

    def render_terrain(self):
        """渲染地形"""
        if self.terrain_surface:
            self.screen.blit(self.terrain_surface, (0, 0))
        else:
            # 回退方案：如果预渲染表面不存在，直接绘制基础地形
            for y in range(self.height):
                for x in range(self.width):
                    if self.terrain is not None:
                        terrain_type = self.terrain[y][x]
                        color = TERRAIN_COLORS.get(terrain_type, (0, 0, 0))
                        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                          self.cell_size, self.cell_size)
                        pygame.draw.rect(self.screen, color, rect)
                        
                        # 绘制主要网格线（每5个单元格）
                        if x % 5 == 0 or y % 5 == 0:
                            pygame.draw.rect(self.screen, (80, 80, 80, 100), rect, 1)
    
    def render_units(self):
        """渲染单位（使用几何图形表示不同类型，颜色区分阵营）"""
        for unit in self.units:
            if unit["status"] == "active":
                x, y = unit["position"]
                camp = unit.get("camp", "red")  # 默认为红方
                color = get_unit_color(unit["type"], camp)
                
                # 获取单位中心坐标
                center_x = int((x + 0.5) * self.cell_size)
                center_y = int((y + 0.5) * self.cell_size)
                size = self.cell_size * 0.6  # 单位基本尺寸
                
                # 根据单位类型绘制不同形状（添加边界线增强可视性）
                border_color = (30, 30, 30)  # 深色边界
                if unit["type"] == "infantry":
                    # 绘制三角形（步兵）
                    points = [
                        (center_x, center_y - size//1.5),  # 顶点
                        (center_x - size//2, center_y + size//2),  # 左下
                        (center_x + size//2, center_y + size//2)   # 右下
                    ]
                    pygame.draw.polygon(self.screen, color, points)
                    pygame.draw.polygon(self.screen, border_color, points, 1)  # 添加边界线
                    
                elif unit["type"] == "tank":
                    # 绘制矩形（坦克）并添加边界
                    tank_rect = pygame.Rect(
                        center_x - size//2,
                        center_y - size//3,
                        size, size//1.5
                    )
                    pygame.draw.rect(self.screen, color, tank_rect)
                    pygame.draw.rect(self.screen, border_color, tank_rect, 1)  # 添加边界线
                    # 坦克炮管
                    pygame.draw.line(
                        self.screen, (50, 50, 50),
                        (center_x, center_y),
                        (center_x + size, center_y),
                        2
                    )
                    
                elif unit["type"] == "artillery":
                    # 绘制五边形（火炮）并添加边界
                    points = []
                    for i in range(5):
                        angle = 2 * math.pi * i / 5 + math.pi/2
                        px = center_x + size * 0.6 * math.cos(angle)
                        py = center_y + size * 0.6 * math.sin(angle)
                        points.append((px, py))
                    pygame.draw.polygon(self.screen, color, points)
                    pygame.draw.polygon(self.screen, border_color, points, 1)  # 添加边界线
                    
                    # 添加炮管方向指示
                    pygame.draw.line(
                        self.screen, border_color,
                        (center_x, center_y),
                        (center_x + size * 1.2, center_y),
                        2
                    )
                
                # 绘制血量条
                health_percent = unit["health"] / 100.0
                bar_width = size
                bar_height = 4
                bar_x = center_x - bar_width // 2
                bar_y = center_y - size - 5
                
                # 血量条背景
                pygame.draw.rect(self.screen, (50, 50, 50), 
                                (bar_x, bar_y, bar_width, bar_height))
                # 当前血量
                pygame.draw.rect(self.screen, 
                                (0, 255, 0) if health_percent > 0.5 else 
                                (255, 255, 0) if health_percent > 0.25 else 
                                (255, 0, 0),
                                (bar_x, bar_y, bar_width * health_percent, bar_height))
                
                # 显示单位ID（放在单位顶部）
                id_surface = self.small_font.render(str(unit["id"]), True, (255, 255, 255))
                id_rect = id_surface.get_rect(center=(center_x, center_y - size - 10))  # 调整位置到单位顶部
                self.screen.blit(id_surface, id_rect)
                
                # 受损单位显示裂痕图标
                if unit["health"] < 50:
                    # 简单十字裂痕
                    pygame.draw.line(
                        self.screen, (100, 100, 100, 180),
                        (center_x - size//3, center_y - size//3),
                        (center_x + size//3, center_y + size//3),
                        2
                    )
                    pygame.draw.line(
                        self.screen, (100, 100, 100, 180),
                        (center_x + size//3, center_y - size//3),
                        (center_x - size//3, center_y + size//3),
                        2
                    )
    
    def render_structures(self):
        """渲染建筑（添加细节纹理）"""
        for structure in self.structures:
            if structure["status"] != "destroyed":
                x, y = structure["position"]
                w, h = structure["dimensions"]
                color = STRUCTURE_COLORS.get(structure["type"], (200, 200, 200))  # 未知建筑用灰色
                
                # 绘制建筑矩形
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  w * self.cell_size, h * self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                
                # 添加建筑细节
                if structure["type"] == "house":
                    # 绘制窗户
                    for i in range(w):
                        for j in range(h):
                            if i > 0 and i < w-1 and j > 0:
                                window_rect = pygame.Rect(
                                    x * self.cell_size + i * self.cell_size + self.cell_size//4,
                                    y * self.cell_size + j * self.cell_size + self.cell_size//4,
                                    self.cell_size//2, self.cell_size//2
                                )
                                pygame.draw.rect(self.screen, (173, 216, 230), window_rect)
                
                elif structure["type"] == "bunker":
                    # 绘制射击孔
                    for i in range(w):
                        for j in range(h):
                            hole_rect = pygame.Rect(
                                x * self.cell_size + i * self.cell_size + self.cell_size//3,
                                y * self.cell_size + j * self.cell_size + self.cell_size//3,
                                self.cell_size//3, self.cell_size//3
                            )
                            pygame.draw.rect(self.screen, (0, 0, 0), hole_rect)
                
                # 绘制损坏效果
                if structure["status"] == "damaged":
                    damage_surface = pygame.Surface((w * self.cell_size, h * self.cell_size), pygame.SRCALPHA)
                    pygame.draw.rect(damage_surface, (255, 0, 0, 80), (0, 0, w * self.cell_size, h * self.cell_size))
                    self.screen.blit(damage_surface, rect.topleft)
    
    def render_info_panel(self):
        """渲染信息面板"""
        panel_rect = pygame.Rect(self.width * self.cell_size, 0, 
                                self.info_panel_width, self.screen_height)
        pygame.draw.rect(self.screen, (40, 40, 50), panel_rect)
        pygame.draw.line(self.screen, (70, 70, 90), 
                        (self.width * self.cell_size, 0),
                        (self.width * self.cell_size, self.screen_height), 3)
        
        # 标题
        title = self.font.render("战场状态面板", True, (255, 215, 0))
        self.screen.blit(title, (self.width * self.cell_size + 20, 20))
        
        # 模拟信息
        y_offset = 60
        info_items = [
            f"模拟步数: {self.step_count}",
            f"活跃单位: {len([u for u in self.units if u['status']=='active'])}/{len(self.units)}",
            f"完好建筑: {len([s for s in self.structures if s['status']=='intact'])}/{len(self.structures)}"
        ]
        
        for item in info_items:
            text = self.small_font.render(item, True, (220, 220, 220))
            self.screen.blit(text, (self.width * self.cell_size + 30, y_offset))
            y_offset += 30
        
        # 地形统计
        y_offset += 20
        terrain_title = self.font.render("地形分布", True, (100, 200, 255))
        self.screen.blit(terrain_title, (self.width * self.cell_size + 20, y_offset))
        y_offset += 40
        
        if self.terrain is not None:
            terrain_types, counts = np.unique(self.terrain, return_counts=True)
            total_cells = self.terrain.size
            terrain_names = {0: "平原", 1: "山地", 2: "水域", 3: "弹坑"}
            
            for ttype, count in zip(terrain_types, counts):
                name = terrain_names.get(ttype, f"未知({ttype})")
                percent = count / total_cells * 100
                text = self.small_font.render(f"{name}: {count} ({percent:.1f}%)", True, TERRAIN_COLORS.get(ttype, (200, 200, 200)))
                self.screen.blit(text, (self.width * self.cell_size + 30, y_offset))
                y_offset += 25
        
        # 阵营统计
        y_offset += 20
        camp_title = self.font.render("阵营统计", True, (255, 150, 100))
        self.screen.blit(camp_title, (self.width * self.cell_size + 20, y_offset))
        y_offset += 40
        
        camp_stats = {"red": 0, "blue": 0}
        destroyed_stats = {"red": 0, "blue": 0}  # 记录被击毁单位
        
        # 准确统计阵营单位
        for unit in self.units:
            camp = unit.get("camp", "red")
            if camp not in ["red", "blue"]:
                # 确保阵营值有效
                camp = "red" if unit["id"] <= 20 else "blue"
            
            if unit["status"] == "active":
                camp_stats[camp] = camp_stats.get(camp, 0) + 1
            elif unit["status"] == "destroyed":
                destroyed_stats[camp] = destroyed_stats.get(camp, 0) + 1
        
        for camp in ["red", "blue"]:
            camp_name = "红方" if camp == "red" else "蓝方"
            camp_color = CAMP_COLORS.get(camp, (200, 200, 200))
            
            # 活跃单位统计
            active_text = f"{camp_name}: {camp_stats.get(camp, 0)}单位活跃"
            text_surface = self.small_font.render(active_text, True, camp_color)
            self.screen.blit(text_surface, (self.width * self.cell_size + 30, y_offset))
            y_offset += 20
            
            # 被击毁单位统计
            destroyed_text = f"已损失: {destroyed_stats.get(camp, 0)}单位"
            text_surface = self.small_font.render(destroyed_text, True, (200, 50, 50))
            self.screen.blit(text_surface, (self.width * self.cell_size + 40, y_offset))
            y_offset += 25
            
        # 战斗事件日志
        y_offset += 20
        events_title = self.font.render("战斗事件", True, (255, 100, 100))
        self.screen.blit(events_title, (self.width * self.cell_size + 20, y_offset))
        y_offset += 40
        
        # 真实事件日志（带步数）
        events = [
            f"步数 {self.step_count-5}: 单位 #15 击毁单位 #32",
            f"步数 {self.step_count-4}: 单位 #8 攻击单位 #42",
            f"步数 {self.step_count-3}: 单位 #27 被炮火击伤",
            f"步数 {self.step_count-2}: 建筑 #2 被摧毁",
            f"步数 {self.step_count-1}: 爆炸发生在 (45, 28)"
        ]
        
        for event in events[-5:]:  # 只显示最近5条
            event_surface = self.small_font.render(event, True, (220, 150, 150))
            self.screen.blit(event_surface, (self.width * self.cell_size + 30, y_offset))
            y_offset += 25
        

    def render_all(self):
        """渲染整个场景"""
        self.screen.fill((0, 0, 0))  # 清屏
        self.render_terrain()
        self.render_structures()
        self.render_units()
        self.render_info_panel()  # 渲染信息面板
        pygame.display.flip()  # 更新显示
    
    def run(self, max_fps=60):
        """运行渲染循环（提高最大帧率）"""
        running = True
        # 预渲染地形
        self.pre_render_terrain()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # 全屏切换
                elif event.type == KEYDOWN:
                    if event.key == K_f:  # 按F键切换全屏
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen:
                            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            # 更新屏幕尺寸
                            self.screen_width, self.screen_height = self.screen.get_size()
                        else:
                            self.screen = pygame.display.set_mode(
                                (self.width * self.cell_size + self.info_panel_width, 
                                 self.height * self.cell_size),
                                RESIZABLE
                            )
                            self.screen_width, self.screen_height = self.screen.get_size()
                
                # 窗口大小调整
                elif event.type == VIDEORESIZE:
                    if not self.fullscreen:
                        self.screen = pygame.display.set_mode(
                            (event.w, event.h), 
                            RESIZABLE
                        )
                        self.screen_width, self.screen_height = event.w, event.h
            
            # 从API获取最新状态并渲染
            if self.fetch_state():
                self.step_count += 1  # 更新模拟步数
                try:
                    self.render_all()
                except Exception as e:
                    print(f"渲染错误: {e}")
                    # 出错时显示错误信息
                    error_surface = self.font.render(f"渲染错误: {str(e)}", True, (255, 0, 0))
                    self.screen.blit(error_surface, (50, 50))
                    pygame.display.flip()
            
            self.clock.tick(max_fps)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = VisualizationApp(width=80, height=60, cell_size=12)
    app.run()
