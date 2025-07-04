# 战区可视化渲染引擎
import pygame
import numpy as np
import math
from entities.unit import Unit
from entities.structure import Structure

# 信息面板配置
INFO_PANEL_WIDTH = 300
PANEL_BG_COLOR = (30, 30, 40, 220)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (70, 130, 180)

# 地形颜色映射
TERRAIN_COLORS = {
    0: (210, 180, 140),  # 平原: 沙色
    1: (139, 137, 137),  # 山地: 灰色
    2: (64, 164, 223),   # 水域: 蓝色
    3: (105, 105, 105)   # 弹坑: 深灰色
}

# 单位颜色映射
UNIT_COLORS = {
    "infantry": (0, 128, 0),    # 步兵: 绿色
    "tank": (128, 0, 0),        # 坦克: 红色
    "artillery": (128, 128, 0), # 火炮: 黄色
    "engineer": (128, 0, 128)   # 工兵: 紫色
}

# 建筑颜色映射
STRUCTURE_COLORS = {
    "house": (200, 150, 100),   # 房屋: 棕色
    "bunker": (80, 80, 80),     # 碉堡: 深灰色
    "bridge": (139, 69, 19)     # 桥梁: 褐色
}

class Renderer:
    def __init__(self, width=100, height=100, cell_size=8):
        """初始化渲染器"""
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.map_width = width * cell_size
        self.map_height = height * cell_size
        self.screen_width = self.map_width + INFO_PANEL_WIDTH
        self.screen_height = self.map_height
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("战区模拟器")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("SimHei", 24)
        self.small_font = pygame.font.SysFont("SimHei", 18)
        
        self.info_messages = []
        self.show_vision = False
        self.vision_surface = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)

    def render_vision_overlay(self, units):
        """渲染同阵营共享视野"""
        for unit in units:
            if unit.is_active():
                x, y = unit.position
                center = (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size))
                radius = unit.vision_range * self.cell_size
                pygame.draw.circle(self.vision_surface, (*unit.camp_color, 50), center, radius)

    def render_terrain(self, terrain_map):
        """渲染地形"""
        for y in range(self.height):
            for x in range(self.width):
                terrain_type = terrain_map[y][x]
                color = TERRAIN_COLORS.get(terrain_type, (0, 0, 0))
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

    def render_units(self, units):
        """渲染单位"""
        for unit in units:
            if unit.is_active():
                x, y = unit.position
                color = UNIT_COLORS.get(unit.type, (255, 255, 255))
                center = (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size))
                radius = int(self.cell_size * 0.4)
                pygame.draw.circle(self.screen, color, center, radius)
                id_surface = self.small_font.render(str(unit.id), True, (255, 255, 255))
                self.screen.blit(id_surface, (x * self.cell_size + 2, y * self.cell_size + 2))

    def render_structures(self, structures):
        """渲染建筑"""
        for structure in structures:
            if not structure.is_destroyed():
                x, y = structure.position
                w, h = structure.dimensions
                color = STRUCTURE_COLORS.get(structure.type, (200, 200, 200))
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 w * self.cell_size, h * self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

    def render_all(self, terrain_map, units, structures):
        """渲染整个场景"""
        self.screen.fill((0, 0, 0))
        self.render_terrain(terrain_map)
        self.render_structures(structures)
        self.render_units(units)
        if self.show_vision:
            self.screen.blit(self.vision_surface, (0,0))
        self.render_info_panel(units, structures)
        pygame.display.flip()

    def render_explosion(self, x, y, radius, intensity):
        """渲染爆炸特效"""
        center = (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size))
        max_radius = int(radius * self.cell_size * intensity)
        pygame.draw.circle(self.screen, (255, 165, 0), center, max_radius//2)
        for i in range(1, 4):
            alpha = 200 - i*50
            wave_radius = max_radius + i*5
            surface = pygame.Surface((wave_radius*2, wave_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (255, 69, 0, alpha), (wave_radius, wave_radius), wave_radius)
            self.screen.blit(surface, (center[0]-wave_radius, center[1]-wave_radius))

    def render_info_panel(self, units, structures):
        """渲染信息面板"""
        panel = pygame.Surface((INFO_PANEL_WIDTH, self.screen_height), pygame.SRCALPHA)
        panel.fill(PANEL_BG_COLOR)
        title = self.font.render("战场态势", True, (255, 215, 0))
        panel.blit(title, (20, 20))
        red = sum(1 for u in units if u.camp == "red" and u.is_active())
        blue = sum(1 for u in units if u.camp == "blue" and u.is_active())
        forces = self.small_font.render(f"红方兵力: {red}  蓝方兵力: {blue}", True, TEXT_COLOR)
        panel.blit(forces, (20, 60))
        self.screen.blit(panel, (self.map_width, 0))
