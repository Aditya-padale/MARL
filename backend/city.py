"""
city.py — Procedurally generated Indian city for the MARL simulator.

1200×900 pixel canvas divided into named zones:
  Residential (top-left), Commercial (top-right), Business District (bottom-right),
  Investment (center), Government (bottom-left), Hospital (top-center),
  Education (left-center), Outskirts/Farmland (far left).

Buildings are procedurally placed within zones using a seeded RNG
for reproducibility. Agents navigate between zones using a simple
waypoint graph — no full A* needed for this scale.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import random
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from config import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    CITY_ZONES, ZoneConfig,
    BUILDING_TEMPLATES, BuildingConfig, CITY_GENERATION_SEED,
    ROAD_WAYPOINTS, ROAD_PROXIMITY_THRESHOLD,
    ROAD_WIDTH, ROAD_COLOR, ROAD_LINE_COLOR,
    AgentType, Action, AGENT_CONFIGS,
    SKY_COLORS, STEPS_PER_MONTH,
)


# ═══════════════════════════════════════════
# BUILDING
# ═══════════════════════════════════════════

@dataclass
class Building:
    """
    A single building rendered on the city canvas.
    Procedurally generated within a zone's boundaries.
    """
    x: int                   # Top-left x coordinate
    y: int                   # Top-left y coordinate
    width: int               # Building width in pixels
    height: int              # Building height in pixels
    color: str               # Fill color (hex)
    label: str               # Display label (e.g., "Shop", "Office")
    zone_name: str           # Parent zone name
    building_id: int         # Unique building ID

    # Visual details (generated)
    has_windows: bool = True
    window_color: str = "#FFD700"    # Warm yellow glow
    roof_color: str = ""             # Slightly darker than building
    door_x: int = 0                  # Door position relative to building
    door_y: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for frontend rendering."""
        return {
            "id": self.building_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": self.color,
            "roof_color": self.roof_color or self._darken_color(self.color, 0.7),
            "label": self.label,
            "zone": self.zone_name,
            "has_windows": self.has_windows,
            "window_color": self.window_color,
            "door_x": self.door_x,
            "door_y": self.door_y,
        }

    @staticmethod
    def _darken_color(hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor (0.0 = black, 1.0 = unchanged)."""
        hex_color = hex_color.lstrip("#")
        r = int(int(hex_color[0:2], 16) * factor)
        g = int(int(hex_color[2:4], 16) * factor)
        b = int(int(hex_color[4:6], 16) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"


# ═══════════════════════════════════════════
# ZONE
# ═══════════════════════════════════════════

@dataclass
class Zone:
    """
    A rectangular zone on the city canvas with buildings and waypoints.
    """
    name: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    waypoints: List[Tuple[int, int]]
    buildings: List[Building] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this zone."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def random_point(self, rng: random.Random) -> Tuple[float, float]:
        """Get a random point within this zone (with margin from edges)."""
        margin = 15
        x = rng.uniform(self.x1 + margin, self.x2 - margin)
        y = rng.uniform(self.y1 + margin, self.y2 - margin)
        return (x, y)

    def random_waypoint(self, rng: random.Random) -> Tuple[float, float]:
        """Get a random waypoint within this zone."""
        if self.waypoints:
            wp = rng.choice(self.waypoints)
            # Add small jitter so agents don't stack exactly on waypoints
            jx = wp[0] + rng.uniform(-10, 10)
            jy = wp[1] + rng.uniform(-10, 10)
            return (jx, jy)
        return self.random_point(rng)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for frontend rendering."""
        return {
            "name": self.name,
            "label": self.label,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "color": self.color,
            "waypoints": self.waypoints,
            "buildings": [b.to_dict() for b in self.buildings],
        }


# ═══════════════════════════════════════════
# WAYPOINT GRAPH
# ═══════════════════════════════════════════

class WaypointGraph:
    """
    Simple navigation graph connecting city zones via road waypoints.

    Agents don't use full A* — they just pathfind through a sparse
    graph of waypoints connected by proximity. For 100 agents on a
    1200×900 canvas, this is more than sufficient.

    Pathfinding: BFS on the waypoint graph to find the shortest
    waypoint path, then agents lerp between waypoints sequentially.
    """

    def __init__(self):
        self.waypoints: List[Tuple[int, int]] = []
        self.adjacency: Dict[int, List[int]] = {}  # wp_index → [neighbor_indices]

    def build(self, waypoints: List[Tuple[int, int]],
              proximity_threshold: int = ROAD_PROXIMITY_THRESHOLD):
        """
        Build the waypoint graph from a list of waypoint coordinates.
        Two waypoints are connected if they're within proximity_threshold pixels.

        Args:
            waypoints:           List of (x, y) coordinates
            proximity_threshold: Max distance for two waypoints to be connected
        """
        self.waypoints = list(waypoints)
        self.adjacency = {i: [] for i in range(len(waypoints))}

        # Connect waypoints within proximity threshold
        for i in range(len(waypoints)):
            for j in range(i + 1, len(waypoints)):
                dist = self._distance(waypoints[i], waypoints[j])
                if dist <= proximity_threshold:
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)

    def find_nearest_waypoint(self, x: float, y: float) -> int:
        """Find the index of the nearest waypoint to a given position."""
        best_idx = 0
        best_dist = float("inf")
        for i, (wx, wy) in enumerate(self.waypoints):
            d = self._distance((x, y), (wx, wy))
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def find_path(self, start_x: float, start_y: float,
                  end_x: float, end_y: float) -> List[Tuple[int, int]]:
        """
        Find the shortest path between two points using BFS on the waypoint graph.

        Returns a list of (x, y) waypoint coordinates forming the path.
        If no path exists, returns a direct path [start → end].

        Args:
            start_x, start_y: Starting position
            end_x, end_y:     Target position

        Returns:
            List of (x, y) waypoints from start to end
        """
        if not self.waypoints:
            return [(int(end_x), int(end_y))]

        start_wp = self.find_nearest_waypoint(start_x, start_y)
        end_wp = self.find_nearest_waypoint(end_x, end_y)

        if start_wp == end_wp:
            return [(int(end_x), int(end_y))]

        # BFS
        visited = {start_wp}
        queue = [(start_wp, [start_wp])]

        while queue:
            current, path = queue.pop(0)

            if current == end_wp:
                # Convert waypoint indices to coordinates
                coords = [self.waypoints[i] for i in path]
                coords.append((int(end_x), int(end_y)))
                return coords

            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found — go direct
        return [(int(end_x), int(end_y))]

    def get_road_segments(self) -> List[Dict[str, Any]]:
        """
        Get all road segments for frontend rendering.

        Returns list of dicts with start/end coordinates for drawing
        road lines on canvas.
        """
        segments = []
        seen = set()

        for i, neighbors in self.adjacency.items():
            for j in neighbors:
                edge = (min(i, j), max(i, j))
                if edge not in seen:
                    seen.add(edge)
                    segments.append({
                        "x1": self.waypoints[i][0],
                        "y1": self.waypoints[i][1],
                        "x2": self.waypoints[j][0],
                        "y2": self.waypoints[j][1],
                    })

        return segments

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)


# ═══════════════════════════════════════════
# CITY
# ═══════════════════════════════════════════

class City:
    """
    The complete city environment: zones, buildings, roads, and navigation.

    Generated procedurally from config.py definitions using a seeded RNG
    for perfect reproducibility (same seed = same city layout every time).

    Responsibilities:
      - Generate buildings within zones
      - Provide navigation targets for agent actions
      - Find nearby agents for social interactions
      - Compute sky color for day/night cycle
      - Serialize full city state for WebSocket
    """

    def __init__(self, seed: int = CITY_GENERATION_SEED):
        """
        Initialize and generate the city.

        Args:
            seed: Random seed for reproducible city generation
        """
        self.seed = seed
        self.rng = random.Random(seed)

        # Build zones from config
        self.zones: Dict[str, Zone] = {}
        self._build_zones()

        # Generate buildings within zones
        self.buildings: List[Building] = []
        self._generate_buildings()

        # Build road/waypoint graph
        self.graph = WaypointGraph()
        self._build_road_network()

        # Zone-to-action mapping for agent navigation
        self._action_zone_map = self._build_action_zone_map()

    # ─────────────────────────────────────────
    # CITY GENERATION
    # ─────────────────────────────────────────

    def _build_zones(self):
        """Create Zone objects from config definitions."""
        for name, cfg in CITY_ZONES.items():
            self.zones[name] = Zone(
                name=cfg.name,
                label=cfg.label,
                x1=cfg.bounds[0],
                y1=cfg.bounds[1],
                x2=cfg.bounds[2],
                y2=cfg.bounds[3],
                color=cfg.color,
                waypoints=list(cfg.waypoints),
            )

    def _generate_buildings(self):
        """
        Procedurally generate buildings within each zone.

        Uses seeded RNG for reproducibility. Buildings are placed with
        collision avoidance (no overlapping) and type-appropriate templates.
        """
        building_id = 0

        # Zone → building template mapping
        zone_template_map = {
            "residential":       ["residential_poor", "residential_mid", "residential_rich"],
            "commercial":        ["commercial_shop", "commercial_market"],
            "business_district": ["business_office", "business_factory"],
            "investment":        ["investment_bank"],
            "government":        ["government_building"],
            "hospital":          ["hospital_building"],
            "education":         ["education_building"],
        }

        for zone_name, zone in self.zones.items():
            cfg = CITY_ZONES.get(zone_name)
            if not cfg or cfg.building_count == 0:
                continue

            templates = zone_template_map.get(zone_name, ["residential_mid"])
            placed_rects: List[Tuple[int, int, int, int]] = []  # For collision check

            for _ in range(cfg.building_count):
                # Pick a random template
                template_name = self.rng.choice(templates)
                template = BUILDING_TEMPLATES.get(template_name)
                if not template:
                    continue

                # Generate building dimensions
                w = self.rng.randint(template.min_width, template.max_width)
                h = self.rng.randint(template.min_height, template.max_height)

                # Try to place without collision (max 20 attempts)
                placed = False
                for _ in range(20):
                    margin = 8
                    bx = self.rng.randint(zone.x1 + margin, max(zone.x2 - w - margin, zone.x1 + margin + 1))
                    by = self.rng.randint(zone.y1 + margin, max(zone.y2 - h - margin, zone.y1 + margin + 1))

                    # Check collision with existing buildings
                    collision = False
                    for (rx, ry, rw, rh) in placed_rects:
                        if (bx < rx + rw + 4 and bx + w + 4 > rx and
                                by < ry + rh + 4 and by + h + 4 > ry):
                            collision = True
                            break

                    if not collision:
                        # Determine window color based on zone type
                        if zone_name == "hospital":
                            window_color = "#FFFFFF"   # White — sterile
                        elif zone_name == "government":
                            window_color = "#FFFFCC"   # Warm institutional
                        elif zone_name == "business_district":
                            window_color = "#87CEEB"   # Blue-tinted glass
                        elif zone_name == "commercial":
                            window_color = "#FFD700"   # Warm shop glow
                        else:
                            window_color = "#FFD700"   # Default warm glow

                        building = Building(
                            x=bx,
                            y=by,
                            width=w,
                            height=h,
                            color=template.color,
                            label=template.label or "",
                            zone_name=zone_name,
                            building_id=building_id,
                            has_windows=(w > 20 and h > 20),
                            window_color=window_color,
                            door_x=bx + w // 2,
                            door_y=by + h,
                        )

                        self.buildings.append(building)
                        zone.buildings.append(building)
                        placed_rects.append((bx, by, w, h))
                        building_id += 1
                        placed = True
                        break

    def _build_road_network(self):
        """
        Build the waypoint navigation graph from config road waypoints
        combined with zone-internal waypoints.
        """
        # Start with road waypoints from config
        all_waypoints = list(ROAD_WAYPOINTS)

        # Add zone waypoints to the global graph
        for zone in self.zones.values():
            for wp in zone.waypoints:
                if wp not in all_waypoints:
                    all_waypoints.append(wp)

        self.graph.build(all_waypoints, ROAD_PROXIMITY_THRESHOLD)

    def _build_action_zone_map(self) -> Dict[Tuple[int, int], List[str]]:
        """
        Build mapping: (action, agent_type) → list of zone names.

        This determines WHERE an agent walks when it takes a particular action.
        E.g., a Farmer taking TRADE goes to "outskirts" (farmland),
        while a Salaried Mid taking TRADE goes to "business_district".
        """
        # Default zone assignments per action
        action_zones = {}

        for agent_type_val, cfg in AGENT_CONFIGS.items():
            # SAVE → go home (residential sub-zone)
            action_zones[(Action.SAVE, agent_type_val)] = [cfg.home_zone]

            # SPEND → go to commercial zone
            action_zones[(Action.SPEND, agent_type_val)] = ["commercial"]

            # INVEST → go to investment zone (bank)
            action_zones[(Action.INVEST, agent_type_val)] = ["investment"]

            # TRADE → go to work zone
            if cfg.work_zone == "none":
                # Seniors: wander between hospital, investment, commercial
                action_zones[(Action.TRADE, agent_type_val)] = [
                    "hospital", "investment", "commercial"
                ]
            else:
                action_zones[(Action.TRADE, agent_type_val)] = [cfg.work_zone]

        return action_zones

    # ─────────────────────────────────────────
    # AGENT NAVIGATION
    # ─────────────────────────────────────────

    def get_target_for_action(self, action: int, agent_type: int) -> Tuple[float, float]:
        """
        Get a target position for an agent based on their action and type.

        Agents walk to the appropriate zone when they take an action:
          SAVE   → home zone (residential)
          SPEND  → commercial zone (shops)
          INVEST → investment zone (bank)
          TRADE  → work zone (type-dependent)

        Args:
            action:     Action enum value (0–3)
            agent_type: AgentType enum value (0–9)

        Returns:
            (x, y) target position within the appropriate zone
        """
        zone_names = self._action_zone_map.get((action, agent_type), ["commercial"])
        zone_name = self.rng.choice(zone_names)

        zone = self.zones.get(zone_name)
        if zone:
            return zone.random_waypoint(self.rng)

        # Fallback: center of canvas
        return (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)

    def get_home_position(self, agent_type: int) -> Tuple[float, float]:
        """
        Get a home position for an agent based on their type.

        Used for:
          - Initial agent placement
          - Nighttime return

        Args:
            agent_type: AgentType enum value

        Returns:
            (x, y) position within the agent's home zone
        """
        cfg = AGENT_CONFIGS.get(agent_type)
        if cfg:
            zone = self.zones.get(cfg.home_zone)
            if zone:
                return zone.random_waypoint(self.rng)

        # Fallback: center of residential zone
        res = self.zones.get("residential")
        if res:
            return res.random_waypoint(self.rng)
        return (300.0, 225.0)

    def get_work_position(self, agent_type: int) -> Tuple[float, float]:
        """
        Get a work position for an agent based on their type.

        Args:
            agent_type: AgentType enum value

        Returns:
            (x, y) position within the agent's work zone
        """
        cfg = AGENT_CONFIGS.get(agent_type)
        if cfg and cfg.work_zone != "none":
            zone = self.zones.get(cfg.work_zone)
            if zone:
                return zone.random_waypoint(self.rng)

        # For types with no work zone (seniors), pick hospital or bank
        fallback_zones = ["hospital", "investment", "commercial"]
        zone_name = self.rng.choice(fallback_zones)
        zone = self.zones.get(zone_name)
        if zone:
            return zone.random_waypoint(self.rng)
        return (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)

    def get_path(self, start_x: float, start_y: float,
                 end_x: float, end_y: float) -> List[Tuple[int, int]]:
        """
        Get a waypoint path between two positions.

        Args:
            start_x, start_y: Starting position
            end_x, end_y:     Target position

        Returns:
            List of (x, y) waypoints forming the path
        """
        return self.graph.find_path(start_x, start_y, end_x, end_y)

    # ─────────────────────────────────────────
    # PROXIMITY QUERIES
    # ─────────────────────────────────────────

    @staticmethod
    def get_agents_in_radius(agent, radius: float,
                             all_agents: list) -> list:
        """
        Find all agents within a given radius of the target agent.

        Used by the social interaction system to find nearby agents
        for proximity-triggered interactions.

        Uses simple O(n) scan — sufficient for 100 agents.
        For 10,000+ agents we'd need spatial hashing, but that's
        unnecessary at this scale.

        Args:
            agent:      The reference agent
            radius:     Search radius in pixels
            all_agents: List of all Agent objects

        Returns:
            List of Agent objects within radius (excluding self)
        """
        nearby = []
        radius_sq = radius * radius  # Avoid sqrt for performance

        for other in all_agents:
            if other.id == agent.id:
                continue
            dx = other.x - agent.x
            dy = other.y - agent.y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius_sq:
                nearby.append(other)

        return nearby

    @staticmethod
    def distance_between(agent1, agent2) -> float:
        """Euclidean distance between two agents."""
        dx = agent1.x - agent2.x
        dy = agent1.y - agent2.y
        return math.sqrt(dx * dx + dy * dy)

    # ─────────────────────────────────────────
    # DAY/NIGHT CYCLE
    # ─────────────────────────────────────────

    @staticmethod
    def get_sky_color(timestep: int) -> str:
        """
        Compute sky gradient color based on time of day.

        Day cycle within each timestep (1 step = 1 day):
          Dawn  (06:00): #f4a261  — warm orange
          Day   (10:00): #87ceeb  — clear blue
          Dusk  (18:00): #e76f51  — sunset red
          Night (22:00): #1a1a2e  — deep dark

        Since each timestep is a full day, we use the sub-step
        fraction for intra-day rendering on the frontend.
        For backend purposes, we return the dominant sky color
        based on a simple hour mapping.

        Args:
            timestep: Current simulation timestep

        Returns:
            Hex color string for sky background
        """
        # Map timestep to approximate "hour" for visual variety
        # Each day cycles through dawn → day → dusk → night
        day_phase = timestep % 4  # Simple 4-phase cycle

        phases = [
            SKY_COLORS["dawn"],
            SKY_COLORS["day"],
            SKY_COLORS["dusk"],
            SKY_COLORS["night"],
        ]

        return phases[day_phase % len(phases)]

    @staticmethod
    def is_night(timestep: int) -> bool:
        """Check if it's currently nighttime in the simulation."""
        return (timestep % 4) == 3

    @staticmethod
    def get_time_of_day_label(timestep: int) -> str:
        """Get a human-readable time-of-day label."""
        phase = timestep % 4
        labels = ["Dawn", "Day", "Dusk", "Night"]
        return labels[phase]

    @staticmethod
    def get_time_label(timestep: int) -> str:
        """
        Get a human-readable time label for the current timestep.

        Format: "Year Y, Month M, Week W, Day D"

        Args:
            timestep: Current simulation timestep

        Returns:
            Formatted time string
        """
        year = timestep // 365 + 1
        day_of_year = timestep % 365
        month = day_of_year // 30 + 1
        week_of_month = (day_of_year % 30) // 7 + 1
        day = day_of_year % 7 + 1

        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        month_name = month_names[min(month - 1, 11)]

        return f"Year {year}, {month_name} Week {week_of_month}"

    # ─────────────────────────────────────────
    # ZONE QUERIES
    # ─────────────────────────────────────────

    def get_zone_at(self, x: float, y: float) -> Optional[str]:
        """
        Get the name of the zone containing a given point.

        Checks primary zones only (not sub-zones) to avoid ambiguity.

        Args:
            x, y: Position on canvas

        Returns:
            Zone name string, or None if outside all zones
        """
        primary_zones = [
            "residential", "commercial", "business_district",
            "investment", "government", "hospital", "education", "outskirts"
        ]
        for name in primary_zones:
            zone = self.zones.get(name)
            if zone and zone.contains_point(x, y):
                return name
        return None

    def get_zone_buildings(self, zone_name: str) -> List[Building]:
        """Get all buildings in a specific zone."""
        zone = self.zones.get(zone_name)
        return zone.buildings if zone else []

    # ─────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────

    def draw_data(self) -> Dict[str, Any]:
        """
        Generate the full city data structure for the frontend.

        Sent once on WebSocket connection as part of the full_state message.
        Contains everything the frontend needs to render the static city:
        zones, buildings, roads, and visual metadata.

        Returns:
            Dict with keys: zones, buildings, roads, road_config, canvas
        """
        # Serialize primary zones (for zone background rendering)
        primary_zone_names = [
            "residential", "commercial", "business_district",
            "investment", "government", "hospital", "education", "outskirts"
        ]
        zones_data = {}
        for name in primary_zone_names:
            zone = self.zones.get(name)
            if zone:
                zones_data[name] = zone.to_dict()

        # All buildings
        buildings_data = [b.to_dict() for b in self.buildings]

        # Road segments
        road_segments = self.graph.get_road_segments()

        # Waypoint positions (for debug rendering)
        waypoints_data = [
            {"x": wp[0], "y": wp[1], "index": i}
            for i, wp in enumerate(self.graph.waypoints)
        ]

        return {
            "canvas": {
                "width": CANVAS_WIDTH,
                "height": CANVAS_HEIGHT,
            },
            "zones": zones_data,
            "buildings": buildings_data,
            "roads": {
                "segments": road_segments,
                "width": ROAD_WIDTH,
                "color": ROAD_COLOR,
                "line_color": ROAD_LINE_COLOR,
            },
            "waypoints": waypoints_data,
            "sky_colors": SKY_COLORS,
            "building_count": len(self.buildings),
            "zone_count": len(zones_data),
            "road_count": len(road_segments),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of the city for logging."""
        return {
            "seed": self.seed,
            "zones": len(self.zones),
            "buildings": len(self.buildings),
            "waypoints": len(self.graph.waypoints),
            "road_segments": len(self.graph.get_road_segments()),
            "canvas_size": f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
        }
