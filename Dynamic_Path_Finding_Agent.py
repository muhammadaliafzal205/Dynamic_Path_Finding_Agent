import pygame
import heapq
import random
import math
import time
from collections import defaultdict

WINDOW_W, WINDOW_H = 1280, 780
PANEL_W = 260
SIDEBAR_W = 220
GRID_AREA_W = WINDOW_W - PANEL_W - SIDEBAR_W
GRID_AREA_H = WINDOW_H
GRID_ORIGIN_X = SIDEBAR_W
GRID_ORIGIN_Y = 0

BG = (10, 14, 26)
PANEL_BG = (13, 18, 32)
BORDER_COL = (30, 58, 95)
ACCENT = (0, 212, 255)
ACCENT2 = (255, 107, 53)
GREEN = (0, 255, 136)
YELLOW = (255, 215, 0)
RED = (255, 51, 102)
DIM = (74, 96, 128)
TEXT_COL = (200, 216, 232)

CELL_COLORS = {
    "empty": (10, 20, 33),
    "wall": (40, 8, 8),
    "frontier": (50, 35, 0),
    "visited": (5, 30, 30),
    "path": (0, 28, 18),
    "start": (0, 30, 56),
    "goal": (40, 0, 17),
}
CELL_BORDER = {
    "empty": (17, 29, 46),
    "wall": (60, 15, 15),
    "frontier": (120, 90, 0),
    "visited": (0, 80, 80),
    "path": (0, 120, 60),
    "start": (0, 120, 200),
    "goal": (180, 0, 60),
}

FPS = 60


class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        _, _, item = heapq.heappop(self._heap)
        return item

    def __len__(self):
        return len(self._heap)

    def empty(self):
        return len(self._heap) == 0


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.walls = set()

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, r, c):
        return (r, c) not in self.walls

    def neighbors(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                yield nr, nc

    def generate_random(self, density, start, goal):
        self.walls.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (start, goal):
                    continue
                if random.random() < density:
                    self.walls.add((r, c))

    def clear_walls(self):
        self.walls.clear()


def manhattan(r, c, gr, gc):
    return abs(r - gr) + abs(c - gc)


def euclidean(r, c, gr, gc):
    return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)


def astar(grid, start, goal, hfunc):
    pq = PriorityQueue()
    g_cost = defaultdict(lambda: float("inf"))
    parent = {}
    g_cost[start] = 0
    h0 = hfunc(*start, *goal)
    pq.push(h0, (start, 0))
    expanded = set()
    nodes_visited = 0

    while not pq.empty():
        node, g = pq.pop()

        if node in expanded:
            continue
        expanded.add(node)
        nodes_visited += 1

        if node != start and node != goal:
            yield ("visited", node, nodes_visited)

        if node == goal:
            path = _reconstruct(parent, goal)
            cost = g_cost[goal]
            yield ("path", path, cost, nodes_visited)
            return

        for nr, nc in grid.neighbors(*node):
            nb = (nr, nc)
            new_g = g + 1
            if new_g < g_cost[nb]:
                g_cost[nb] = new_g
                parent[nb] = node
                f = new_g + hfunc(nr, nc, *goal)
                pq.push(f, (nb, new_g))
                if nb != goal:
                    yield ("frontier", nb, nodes_visited)

    yield ("fail", None, 0, nodes_visited)


def gbfs(grid, start, goal, hfunc):
    pq = PriorityQueue()
    visited = set()
    parent = {}
    pq.push(hfunc(*start, *goal), (start,))
    visited.add(start)
    nodes_visited = 0

    while not pq.empty():
        (node,) = pq.pop()
        nodes_visited += 1

        if node != start and node != goal:
            yield ("visited", node, nodes_visited)

        if node == goal:
            path = _reconstruct(parent, goal)
            cost = len(path) - 1
            yield ("path", path, cost, nodes_visited)
            return

        for nr, nc in grid.neighbors(*node):
            nb = (nr, nc)
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                pq.push(hfunc(nr, nc, *goal), (nb,))
                if nb != goal:
                    yield ("frontier", nb, nodes_visited)

    yield ("fail", None, 0, nodes_visited)


def _reconstruct(parent, goal):
    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(node)
    path.reverse()
    return path


def draw_rect_border(surf, color, border_color, rect, radius=0):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    pygame.draw.rect(surf, border_color, rect, 1, border_radius=radius)


def draw_text(surf, text, font, color, x, y, align="left"):
    img = font.render(str(text), True, color)
    if align == "center":
        x -= img.get_width() // 2
    elif align == "right":
        x -= img.get_width()
    surf.blit(img, (x, y))


class Button:
    def __init__(self, rect, label, color, hover_color, font, text_color=BG):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.font = font
        self.text_color = text_color
        self.disabled = False

    def draw(self, surf):
        mouse = pygame.mouse.get_pos()
        hov = self.rect.collidepoint(mouse) and not self.disabled
        col = self.hover_color if hov else self.color
        if self.disabled:
            col = (40, 40, 55)
        pygame.draw.rect(surf, col, self.rect, border_radius=3)
        pygame.draw.rect(surf, BORDER_COL, self.rect, 1, border_radius=3)
        img = self.font.render(
            self.label, True, self.text_color if not self.disabled else DIM
        )
        surf.blit(img, img.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and self.rect.collidepoint(event.pos)
            and not self.disabled
        )


class RadioGroup:
    def __init__(self, options, x, y, font, gap=28):
        self.options = options
        self.x, self.y = x, y
        self.font = font
        self.gap = gap
        self.selected = 0

    def draw(self, surf):
        for i, (label, _) in enumerate(self.options):
            ry = self.y + i * self.gap
            active = i == self.selected
            dot_col = ACCENT if active else DIM
            text_col = ACCENT if active else TEXT_COL
            pygame.draw.circle(
                surf, dot_col, (self.x + 8, ry + 8), 6, 0 if active else 2
            )
            draw_text(surf, label, self.font, text_col, self.x + 20, ry)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, _ in enumerate(self.options):
                ry = self.y + i * self.gap
                if pygame.Rect(self.x, ry, 160, self.gap).collidepoint(event.pos):
                    self.selected = i
                    return True
        return False

    @property
    def value(self):
        return self.options[self.selected][1]


class Toggle:
    def __init__(self, x, y, label, font):
        self.x, self.y = x, y
        self.label = label
        self.font = font
        self.on = False

    def draw(self, surf):
        draw_text(surf, self.label, self.font, TEXT_COL, self.x, self.y + 3)
        tx = self.x + 170
        bg = (50, 20, 5) if self.on else (20, 25, 40)
        pygame.draw.rect(surf, bg, (tx, self.y + 2, 36, 16), border_radius=8)
        pygame.draw.rect(
            surf, ACCENT2 if self.on else DIM, (tx, self.y + 2, 36, 16), 1, border_radius=8
        )
        knob_x = tx + 21 if self.on else tx + 3
        knob_col = ACCENT2 if self.on else DIM
        pygame.draw.circle(surf, knob_col, (knob_x + 6, self.y + 10), 6)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            tx = self.x + 170
            if pygame.Rect(tx, self.y, 36, 20).collidepoint(event.pos):
                self.on = not self.on
                return True
        return False


class Slider:
    def __init__(self, x, y, w, min_val, max_val, default, label, font):
        self.rect = pygame.Rect(x, y, w, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.label = label
        self.font = font
        self.dragging = False

    def draw(self, surf):
        draw_text(
            surf, f"{self.label}: {self.value}", self.font, DIM, self.rect.x, self.rect.y - 16
        )
        pygame.draw.rect(surf, BORDER_COL, self.rect, border_radius=5)
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = int(t * self.rect.w)
        pygame.draw.rect(
            surf, ACCENT, (self.rect.x, self.rect.y, fill_w, self.rect.h), border_radius=5
        )
        kx = self.rect.x + fill_w
        pygame.draw.circle(surf, ACCENT, (kx, self.rect.centery), 7)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.inflate(20, 20).collidepoint(
            event.pos
        ):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            t = max(0, min(1, (event.pos[0] - self.rect.x) / self.rect.w))
            self.value = int(self.min_val + t * (self.max_val - self.min_val))
            return True
        return False


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()

        self.f_small = pygame.font.SysFont("consolas", 11)
        self.f_med = pygame.font.SysFont("consolas", 13, bold=True)
        self.f_large = pygame.font.SysFont("consolas", 22, bold=True)
        self.f_title = pygame.font.SysFont("consolas", 14, bold=True)

        self.rows, self.cols = 25, 40
        self.grid = Grid(self.rows, self.cols)
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self._compute_cell_size()

        self.cell_state = {}

        self.search_gen = None
        self.running = False
        self.path = []
        self.nodes_vis = 0
        self.path_cost = 0
        self.exec_ms = 0
        self.replans = 0
        self.start_time = 0
        self.status_msg = "Ready. Press RUN to start."
        self.status_ok = False
        self.log_lines = []

        self.edit_tool = "wall"
        self.mouse_down = False

        self.spawn_timer = 0

        self.steps_per_frame = 5

        sx = 10
        self._build_ui(sx)

        self.grid.generate_random(0.30, self.start, self.goal)
        self._sync_cell_states()

    def _build_ui(self, sx):
        F = self.f_small

        self.algo_radio = RadioGroup([("A* Search", "astar"), ("Greedy BFS", "gbfs")], sx, 80, F)

        self.heur_radio = RadioGroup([("Manhattan", "manhattan"), ("Euclidean", "euclidean")], sx, 190, F)

        self.edit_radio = RadioGroup(
            [("Draw Walls", "wall"), ("Erase", "erase"), ("Set Start", "start"), ("Set Goal", "goal")],
            sx,
            310,
            F,
        )

        self.sl_density = Slider(sx, 460, SIDEBAR_W - 20, 0, 70, 30, "Density %", F)
        self.sl_speed = Slider(sx, 530, SIDEBAR_W - 20, 1, 50, 5, "Steps/Frame", F)
        self.sl_prob = Slider(sx, 600, SIDEBAR_W - 20, 1, 20, 5, "Spawn %", F)

        self.dynamic_toggle = Toggle(sx, 650, "DYNAMIC MODE", F)

        BW, BH = SIDEBAR_W - 20, 26
        self.btn_gen = Button((sx, 700, BW, BH), "GENERATE MAP", (30, 15, 0), YELLOW, self.f_med, YELLOW)
        self.btn_run = Button((sx, 732, BW, BH), "RUN SEARCH", (0, 20, 30), ACCENT, self.f_med, ACCENT)
        self.btn_stop = Button((sx, 732, BW, BH), "STOP", (30, 0, 10), RED, self.f_med, RED)
        self.btn_clear = Button((sx, 760, BW - 1, BH - 2), "CLEAR PATH", (0, 15, 5), GREEN, self.f_small, GREEN)
        self.btn_stop.disabled = True

    def _compute_cell_size(self):
        self.cell_w = GRID_AREA_W // self.cols
        self.cell_h = GRID_AREA_H // self.rows
        self.cell_w = max(8, min(self.cell_w, self.cell_h))
        self.cell_h = self.cell_w
        self.grid_px_w = self.cols * self.cell_w
        self.grid_px_h = self.rows * self.cell_h
        self.grid_ox = GRID_ORIGIN_X + (GRID_AREA_W - self.grid_px_w) // 2
        self.grid_oy = GRID_ORIGIN_Y + (GRID_AREA_H - self.grid_px_h) // 2

    def _cell_from_mouse(self, mx, my):
        c = (mx - self.grid_ox) // self.cell_w
        r = (my - self.grid_oy) // self.cell_h
        return r, c

    def _cell_rect(self, r, c):
        x = self.grid_ox + c * self.cell_w
        y = self.grid_oy + r * self.cell_h
        return pygame.Rect(x, y, self.cell_w, self.cell_h)

    def _sync_cell_states(self):
        self.cell_state.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.start:
                    self.cell_state[(r, c)] = "start"
                elif (r, c) == self.goal:
                    self.cell_state[(r, c)] = "goal"
                elif (r, c) in self.grid.walls:
                    self.cell_state[(r, c)] = "wall"
                else:
                    self.cell_state[(r, c)] = "empty"

    def _clear_path_states(self):
        for key, val in list(self.cell_state.items()):
            if val in ("frontier", "visited", "path"):
                self.cell_state[key] = "wall" if key in self.grid.walls else "empty"

    def _start_search(self, from_pos=None):
        self._clear_path_states()
        self._sync_cell_states()
        start = from_pos or self.start
        hfunc = manhattan if self.heur_radio.value == "manhattan" else euclidean
        if self.algo_radio.value == "astar":
            self.search_gen = astar(self.grid, start, self.goal, hfunc)
        else:
            self.search_gen = gbfs(self.grid, start, self.goal, hfunc)
        self.running = True
        self.path = []
        self.nodes_vis = 0
        self.start_time = time.time()
        self.status_msg = f"Running {self.algo_radio.value.upper()} ({self.heur_radio.value})..."
        self.status_ok = False
        self.btn_run.disabled = True
        self.btn_stop.disabled = False
        self._log(f"Started {self.algo_radio.value} | heuristic={self.heur_radio.value}")

    def _stop_search(self):
        self.running = False
        self.search_gen = None
        self.btn_run.disabled = False
        self.btn_stop.disabled = True

    def _step_search(self):
        if not self.running or self.search_gen is None:
            return
        for _ in range(self.steps_per_frame):
            try:
                event_type, *data = next(self.search_gen)
            except StopIteration:
                self._stop_search()
                return

            if event_type == "frontier":
                node, nv = data[0], data[1]
                if self.cell_state.get(node) not in ("start", "goal"):
                    self.cell_state[node] = "frontier"
                self.nodes_vis = nv

            elif event_type == "visited":
                node, nv = data[0], data[1]
                if self.cell_state.get(node) not in ("start", "goal"):
                    self.cell_state[node] = "visited"
                self.nodes_vis = nv

            elif event_type == "path":
                path, cost, nv = data[0], data[1], data[2]
                self.path = path
                self.path_cost = cost
                self.nodes_vis = nv
                self.exec_ms = (time.time() - self.start_time) * 1000
                for p in path:
                    if self.cell_state.get(p) not in ("start", "goal"):
                        self.cell_state[p] = "path"
                self.status_msg = f"PATH FOUND  cost={cost}  nodes={nv}"
                self.status_ok = True
                self._log(f"Path found! Length={len(path)} Cost={cost} Nodes={nv}", ok=True)
                self._stop_search()
                return

            elif event_type == "fail":
                nv = data[2]
                self.nodes_vis = nv
                self.exec_ms = (time.time() - self.start_time) * 1000
                self.status_msg = "NO PATH FOUND"
                self.status_ok = False
                self._log("No path found!", warn=True)
                self._stop_search()
                return

    def _maybe_spawn_obstacle(self):
        if not self.dynamic_toggle.on or not self.running:
            return
        self.spawn_timer += 1
        if self.spawn_timer < 10:
            return
        self.spawn_timer = 0
        prob = self.sl_prob.value / 100.0
        if random.random() > prob:
            return
        for _ in range(20):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if (r, c) in (self.start, self.goal):
                continue
            if (r, c) in self.grid.walls:
                continue
            self.grid.walls.add((r, c))
            self.cell_state[(r, c)] = "wall"
            if self.path and (r, c) in self.path:
                self.replans += 1
                replan_from = self.path[0]
                self._log(f"Obstacle blocked path! Re-planning #{self.replans}", warn=True)
                self._start_search(from_pos=replan_from)
            break

    def _apply_edit(self, mx, my):
        r, c = self._cell_from_mouse(mx, my)
        if not self.grid.in_bounds(r, c):
            return
        tool = self.edit_radio.value
        if tool == "wall":
            if (r, c) in (self.start, self.goal):
                return
            self.grid.walls.add((r, c))
            self.cell_state[(r, c)] = "wall"
        elif tool == "erase":
            self.grid.walls.discard((r, c))
            self.cell_state[(r, c)] = "empty"
        elif tool == "start":
            old = self.start
            self.start = (r, c)
            self.grid.walls.discard((r, c))
            self.cell_state[old] = "wall" if old in self.grid.walls else "empty"
            self.cell_state[(r, c)] = "start"
        elif tool == "goal":
            old = self.goal
            self.goal = (r, c)
            self.grid.walls.discard((r, c))
            self.cell_state[old] = "wall" if old in self.grid.walls else "empty"
            self.cell_state[(r, c)] = "goal"

    def _draw_grid(self):
        CW, CH = self.cell_w, self.cell_h
        for (r, c), state in self.cell_state.items():
            rect = self._cell_rect(r, c)
            col = CELL_COLORS[state]
            bdr = CELL_BORDER[state]
            pygame.draw.rect(self.screen, col, rect)
            pygame.draw.rect(self.screen, bdr, rect, 1)

            if state == "path" and CW >= 12:
                inner = rect.inflate(-CW // 2, -CH // 2)
                pygame.draw.ellipse(self.screen, (0, 180, 80), inner)
            elif state == "frontier" and CW >= 10:
                inner = rect.inflate(-CW // 2, -CH // 2)
                pygame.draw.ellipse(self.screen, (200, 150, 0), inner)
            elif state == "start":
                draw_text(self.screen, "S", self.f_med, ACCENT, rect.x + CW // 2, rect.y + CH // 2 - 7, align="center")
            elif state == "goal":
                draw_text(self.screen, "G", self.f_med, RED, rect.x + CW // 2, rect.y + CH // 2 - 7, align="center")

    def _draw_sidebar(self):
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SIDEBAR_W, WINDOW_H))
        pygame.draw.line(self.screen, BORDER_COL, (SIDEBAR_W, 0), (SIDEBAR_W, WINDOW_H), 1)

        sx = 10
        draw_text(self.screen, "PATHFINDING", self.f_title, ACCENT, sx, 10)
        draw_text(self.screen, "AGENT v1.0", self.f_small, DIM, sx, 28)
        pygame.draw.line(self.screen, BORDER_COL, (sx, 50), (SIDEBAR_W - 10, 50), 1)

        draw_text(self.screen, "ALGORITHM", self.f_small, DIM, sx, 65)
        self.algo_radio.draw(self.screen)

        draw_text(self.screen, "HEURISTIC", self.f_small, DIM, sx, 175)
        self.heur_radio.draw(self.screen)

        draw_text(self.screen, "EDIT TOOL", self.f_small, DIM, sx, 295)
        self.edit_radio.draw(self.screen)

        draw_text(self.screen, "MAP CONFIG", self.f_small, DIM, sx, 435)
        self.sl_density.draw(self.screen)
        self.sl_speed.draw(self.screen)
        self.sl_prob.draw(self.screen)
        self.dynamic_toggle.draw(self.screen)

        self.btn_gen.draw(self.screen)
        if self.running:
            self.btn_stop.draw(self.screen)
        else:
            self.btn_run.draw(self.screen)
        self.btn_clear.draw(self.screen)

    def _draw_metrics(self):
        px = WINDOW_W - PANEL_W
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(self.screen, BORDER_COL, (px, 0), (px, WINDOW_H), 1)

        mx = px + 12
        draw_text(self.screen, "METRICS", self.f_title, ACCENT, mx, 12)
        pygame.draw.line(self.screen, BORDER_COL, (mx, 35), (WINDOW_W - 12, 35), 1)

        metrics = [
            ("NODES VISITED", str(self.nodes_vis), ACCENT),
            ("PATH LENGTH", str(len(self.path)), GREEN),
            ("PATH COST", str(self.path_cost), ACCENT2),
            ("EXEC TIME (ms)", f"{self.exec_ms:.1f}", YELLOW),
            ("RE-PLANS", str(self.replans), RED),
        ]
        y = 50
        for label, val, col in metrics:
            draw_text(self.screen, label, self.f_small, DIM, mx, y)
            draw_text(self.screen, val, self.f_large, col, mx, y + 14)
            pygame.draw.line(self.screen, BORDER_COL, (mx, y + 42), (WINDOW_W - 12, y + 42), 1)
            y += 52

        y += 6
        draw_text(self.screen, "STATUS", self.f_small, DIM, mx, y)
        col = GREEN if self.status_ok else TEXT_COL
        words = self.status_msg.split()
        line, lines = "", []
        for w in words:
            test = line + w + " "
            if self.f_small.size(test)[0] > PANEL_W - 24:
                lines.append(line)
                line = w + " "
            else:
                line = test
        lines.append(line)
        for i, ln in enumerate(lines[:3]):
            draw_text(self.screen, ln, self.f_small, col, mx, y + 16 + i * 16)

        y = WINDOW_H - 220
        draw_text(self.screen, "LEGEND", self.f_small, DIM, mx, y)
        legend = [
            ("empty", "Empty Cell"),
            ("wall", "Wall"),
            ("start", "Start Node"),
            ("goal", "Goal Node"),
            ("frontier", "Frontier (Queue)"),
            ("visited", "Visited (Expanded)"),
            ("path", "Final Path"),
        ]
        y += 18
        for state, label in legend:
            pygame.draw.rect(self.screen, CELL_COLORS[state], (mx, y, 14, 14))
            pygame.draw.rect(self.screen, CELL_BORDER[state], (mx, y, 14, 14), 1)
            draw_text(self.screen, label, self.f_small, DIM, mx + 20, y + 1)
            y += 20

        draw_text(self.screen, "LOG", self.f_small, DIM, mx, y + 4)
        y += 20
        for line, ok, warn in self.log_lines[-8:]:
            col = GREEN if ok else (YELLOW if warn else DIM)
            draw_text(self.screen, line, self.f_small, col, mx, y)
            y += 15

    def _log(self, msg, ok=False, warn=False):
        self.log_lines.append((msg[:35], ok, warn))
        if len(self.log_lines) > 50:
            self.log_lines.pop(0)

    def run(self):
        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.grid.generate_random(self.sl_density.value / 100, self.start, self.goal)
                        self._sync_cell_states()
                    elif event.key == pygame.K_SPACE and not self.running:
                        self._start_search()
                    elif event.key == pygame.K_ESCAPE:
                        self._stop_search()
                    elif event.key == pygame.K_c:
                        self._clear_path_states()

                self.sl_density.handle(event)
                self.sl_speed.handle(event)
                self.sl_prob.handle(event)
                self.steps_per_frame = max(1, self.sl_speed.value)

                self.dynamic_toggle.handle(event)

                self.algo_radio.handle(event)
                self.heur_radio.handle(event)
                self.edit_radio.handle(event)

                if self.btn_gen.clicked(event):
                    self._stop_search()
                    self.grid.generate_random(self.sl_density.value / 100, self.start, self.goal)
                    self._sync_cell_states()
                    self._log("Map generated", ok=True)

                if not self.running and self.btn_run.clicked(event):
                    self._start_search()

                if self.btn_stop.clicked(event):
                    self._stop_search()
                    self.status_msg = "Stopped by user."

                if self.btn_clear.clicked(event):
                    self._stop_search()
                    self._clear_path_states()
                    self.path = []
                    self.nodes_vis = self.path_cost = self.exec_ms = 0
                    self.status_msg = "Path cleared."

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down = True
                    self._apply_edit(*event.pos)
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False
                if event.type == pygame.MOUSEMOTION and self.mouse_down:
                    self._apply_edit(*event.pos)

            self._step_search()
            self._maybe_spawn_obstacle()

            self.screen.fill(BG)
            self._draw_grid()
            self._draw_sidebar()
            self._draw_metrics()

            pygame.display.flip()


if __name__ == "__main__":
    App().run()