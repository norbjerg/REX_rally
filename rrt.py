import numpy as np

from constants import Constants


class PointMassModel:
    
    def __init__(self, ctrl_range) -> None:
        self.ctrl_range = ctrl_range
        return

    def forward_dyn(self, x, u, T):
        path = [x]
        for i in range(T):
            x_new = path[-1] + u[i]
            path.append(x_new)

        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        dir = (x_goal - x) / np.linalg.norm(x_goal - x)

        u = np.array([dir * self.ctrl_range[1] for _ in range(T)])

        return self.forward_dyn(x, u, T)


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, pos):
            self.pos = pos  # configuration position, usually 2D/3D for planar robots
            self.path = []  # the path with a integration horizon. this could be just a straight line for holonomic system
            self.parent = None

        def calc_distance_to(self, to_node):
            # node distance can be nontrivial as some form of cost-to-go function for e.g. underactuated system
            # use euclidean norm for basic holonomic point mass or as heuristics
            d = np.linalg.norm(np.array(to_node.pos) - np.array(self.pos))
            return d

    def __init__(
        self,
        start,
        goal,
        map,
        expand_dis=20,
        path_resolution=5,
        goal_sample_rate=5,
        max_iter=500,
    ):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.robot = PointMassModel(Constants.Robot.CTRL_RANGE)
        self.map = map

        self.min_rand = map.map_area[0]
        self.max_rand = map.map_area[1]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.node_list = []

    def planning(self):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision_free(new_node):
                self.node_list.append(new_node)

            # try to steer towards the goal if we are already close enough
            if self.node_list[-1].calc_distance_to(self.end) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision_free(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        # integrate the robot dynamics towards the sampled position
        # for holonomic point pass robot, this could be straight forward as a straight line in Euclidean space
        # while need some local optimization to find the dynamically closest path otherwise
        new_node = self.Node(from_node.pos)
        d = new_node.calc_distance_to(to_node)

        new_node.path = [new_node.pos]

        if extend_length > d:
            extend_length = d

        n_expand = int(extend_length // self.path_resolution)

        if n_expand > 0:
            steer_path = self.robot.inverse_dyn(new_node.pos, to_node.pos, n_expand)
            # use the end position to represent the current node and update the path
            new_node.pos = steer_path[-1]
            new_node.path += steer_path

        d = new_node.calc_distance_to(to_node)
        if d <= self.path_resolution:
            # this is considered as connectable
            new_node.path.append(to_node.pos)

            # so this position becomes the representation of this node
            new_node.pos = to_node.pos.copy()

        new_node.parent = from_node
        return new_node

    def find_n_points(self, a1, a2, n):
        # Convert to numpy arrays for element-wise operations
        a1 = np.array(a1)
        a2 = np.array(a2)

        # Generate n evenly spaced points between 0 and 1 (excluding endpoints)
        t_values = np.linspace(0, 1, n + 2)[1:-1]  # exclude 0 and 1

        # Interpolate between a1 and a2 using the t_values
        points = [(1 - t) * a1 + t * a2 for t in t_values]

        return points

    def increase_point_density(self, node, n=0.0):
        left = np.array([node[0] - n, node[1]])
        right = np.array([node[0] + n, node[1]])
        up = np.array([node[0], node[1] + n])
        down = np.array([node[0], node[1] - n])
        left_up = np.array([node[0] - n, node[1] + n])
        left_down = np.array([node[0] - n, node[1] - n])
        right_up = np.array([node[0] + n, node[1] + n])
        right_down = np.array([node[0] + n, node[1] - n])
        if (
            self.map.in_collision(left)
            or self.map.in_collision(right)
            or self.map.in_collision(up)
            or self.map.in_collision(down)
        ):
            return True
        elif (
            self.map.in_collision(left_up)
            or self.map.in_collision(left_down)
            or self.map.in_collision(right_up)
            or self.map.in_collision(right_down)
        ):
            return True
        else:
            return False

    def shorten_path(self, path):
        current_node = path[0]
        next_node = path[1]
        shortened_path = [current_node]

        for i in range(1, len(path) - 1):
            # increase n for better accuracy
            n = 20
            distance = (np.linalg.norm(np.array(next_node) - np.array(current_node)) + 0.5) * n
            middles = self.find_n_points(current_node, next_node, int(distance))

            collission_free = True
            for p in middles:
                # 0.025 is just how many pixels I want to expand the robot by
                if self.increase_point_density(p):
                    collission_free = False
                    break
            if not collission_free:
                shortened_path.append(path[i - 1])
                current_node = next_node
                next_node = path[i + 1]
            else:
                next_node = path[i + 1]
        shortened_path.append(next_node)
        return shortened_path

    def generate_final_course(self, goal_ind):
        path = [self.end.pos]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.pos)
            node = node.parent
        path.append(node.pos)
        shorten_path = self.shorten_path(path)
        return shorten_path

    def get_random_node(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(np.random.uniform(self.map.map_area[0], self.map.map_area[1]))
        else:  # goal point sampling
            rnd = self.Node(self.end.pos)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [node.calc_distance_to(rnd_node) for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def check_collision_free(self, node):
        if node is None:
            return False
        for p in node.path:
            if self.map.in_collision(np.array(p)):
                return False

        return True


class GridOccupancyMap(object):
    """ """

    def __init__(self, low=(0, 0), high=(500, 500), res=5) -> None:
        self.map_area = [low, high]  # a rectangular area
        self.map_size = np.array([high[0] - low[0], high[1] - low[1]])
        self.resolution = res

        self.n_grids = [int(s // res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [
            self.map_area[0][0],
            self.map_area[1][0],
            self.map_area[0][1],
            self.map_area[1][1],
        ]

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = [int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1

        return self.grid[indices[0], indices[1]]

    def populate(self, obstacles):
        """
        generate a grid map with some circle shaped obstacles
        """

        origins = obstacles
        radius = (Constants.Obstacle.SHAPE_RADIUS) / 1000

        # fill the grids by checking if the grid centroid is in any of the circle
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array(
                    [
                        self.map_area[0][0] + self.resolution * (i + 0.5),
                        self.map_area[0][1] + self.resolution * (j + 0.5),
                    ]
                )
                for o in origins:
                    if np.linalg.norm(centroid - o) <= radius:
                        self.grid[i, j] = 1
                        break
