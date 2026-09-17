from collections import deque


FEATURE_NAMES = [
    "connectivity",
    "accessibility",
    "obstacle_density",
    "connected_components",
    "largest_open_area",
    "border_obstacle_ratio",
    "balance",
]


def extract_feature_dict(grid):
    height, width = len(grid), len(grid[0])
    total = height * width
    open_cells = sum(cell == 0 for row in grid for cell in row)
    visited = set()
    component_sizes = []

    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0 or (x, y) in visited:
                continue
            queue = deque([(x, y)])
            visited.add((x, y))
            size = 0
            while queue:
                current_x, current_y = queue.popleft()
                size += 1
                for next_x, next_y in (
                    (current_x - 1, current_y), (current_x + 1, current_y),
                    (current_x, current_y - 1), (current_x, current_y + 1),
                ):
                    if (0 <= next_x < width and 0 <= next_y < height
                            and grid[next_y][next_x] == 0
                            and (next_x, next_y) not in visited):
                        visited.add((next_x, next_y))
                        queue.append((next_x, next_y))
            component_sizes.append(size)

    largest = max(component_sizes, default=0)
    border = grid[0] + grid[-1] + [row[0] for row in grid[1:-1]] + [row[-1] for row in grid[1:-1]]
    accessibility = open_cells / total
    obstacle_density = 1.0 - accessibility
    return {
        "connectivity": largest / open_cells if open_cells else 0.0,
        "accessibility": accessibility,
        "obstacle_density": obstacle_density,
        "connected_components": len(component_sizes),
        "largest_open_area": largest / total,
        "border_obstacle_ratio": sum(border) / len(border),
        "balance": 1.0 - abs(accessibility - 0.65),
    }


def extract_features(grid):
    features = extract_feature_dict(grid)
    return [features[name] for name in FEATURE_NAMES]
