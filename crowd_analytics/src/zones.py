import yaml
from shapely.geometry import Point, Polygon

class ZoneCounter:
    def __init__(self, zone_config_path):
        with open(zone_config_path, "r") as f:
            data = yaml.safe_load(f)

        self.zones = []
        for z in data["zones"]:
            self.zones.append({
                "name": z["name"],
                "polygon": Polygon(z["polygon"]),  # for point-in-polygon
                "points": z["polygon"]              # for drawing
            })

    def count(self, boxes):
        counts = {z["name"]: 0 for z in self.zones}

        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            point = Point(cx, cy)

            for z in self.zones:
                if z["polygon"].contains(point):
                    counts[z["name"]] += 1

        return counts