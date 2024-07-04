import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


class Triangle:
    def __init__(self, vertex1, vertex2, vertex3):
        self.vertices = tuple(map(int, (vertex1, vertex2, vertex3)))
        self.vertex1, self.vertex2, self.vertex3 = self.vertices

        self.vector = self.calculate_vector()  # Calculate the vector using cross product
        self.center = self.calculate_center()  # Geometric center
        self.pos_plane = []  # List of points on positive side of the plane

    def calculate_vector(self):
        vector1 = pts[self.vertex2] - pts[self.vertex1]
        vector2 = pts[self.vertex3] - pts[self.vertex1]
        normal_vector = np.cross(vector1, vector2)
        norm = np.linalg.norm(normal_vector)
        if norm == 0:
            return normal_vector
        return normal_vector / norm

    def calculate_center(self):
        return (pts[self.vertex1] + pts[self.vertex2] + pts[self.vertex3]) / 3

    def triangle_print(self):
        return f"Triangle: vertex1 = {self.vertex1}, vertex2 = {self.vertex2}, vertex3 = {self.vertex3}"

    def equal_triangles(self, other):
        if not isinstance(other, Triangle):
            return False
        return all(np.array_equal(pts[own], pts[oth]) for own, oth in zip(self.vertices, other.vertices))

    def triangle_hash(self):
        return hash(self.vertices)





def update_triangle_pos_plane(triangles, pts):
    for triangle in triangles:
        if not hasattr(triangle, 'pos_plane'):
            triangle.pos_plane = []

        # Calculate points for each triangle
        positive_points = [
            [i, np.dot(triangle.vector, pts[i] - triangle.center)]
            for i in range(len(pts))
            if i not in [triangle.vertex1, triangle.vertex2, triangle.vertex3] and
               np.dot(triangle.vector, pts[i] - triangle.center) > 0
        ]
        triangle.pos_plane.extend(positive_points)


def quickhull():
    tetrahedron = initial_tetrahedron(pts)

    # Create the tetrahedron triangles
    triangles = [Triangle(tetrahedron[i], tetrahedron[j], tetrahedron[k])
                 for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]

    # Vectors pointing outward using the opposite point
    for triangle in triangles:
        opposite = next(i for i in tetrahedron if i not in [triangle.vertex1, triangle.vertex2, triangle.vertex3])
        # Check if the normal vector needs to be reversed
        if np.dot(triangle.vector, pts[opposite] - triangle.center) < 0:
            triangle.vector *= -1

    # Define lines from vectors starting in the center and from triangle edges
    pos_plane_center = []
    vectors = [[triangle.center, triangle.center + triangle.vector] for triangle in triangles]
    edges = [[getattr(triangle, pair[0]), getattr(triangle, pair[1])]
             for triangle in triangles
             for pair in [('vertex1', 'vertex2'), ('vertex1', 'vertex3'), ('vertex2', 'vertex3')]]

    update_triangle_pos_plane(triangles, pts)

    # Create lines from positive points to triangle center
    for triangle in triangles:
        pos_plane_center.extend([[triangle.center, pts[point[0]]] for point in triangle.pos_plane])

    def plot_points_and_labels(ax, points):
        for i, point in enumerate(points):
            ax.scatter3D(*point, c='r')
            ax.text(*point, str(i))

    def plot_lines(ax, lines, color):
        for line in lines:
            ax.plot3D([line[0][0], line[1][0]],
                      [line[0][1], line[1][1]],
                      [line[0][2], line[1][2]],
                      c=color)

    draw_tetraeder = False
    if draw_tetraeder:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        plot_points_and_labels(ax, [pts[i] for i in range(len(pts))])
        plot_lines(ax, vectors, 'b')
        plot_lines(ax, [[pts[line[0]], pts[line[1]]] for line in edges], 'g')
        plot_lines(ax, pos_plane_center, 'y')

        plt.show()

    hull = []
    list_of_triangles = []

    # Add the triangles to the list (hull)
    for triangle in triangles:
        list_of_triangles.append(triangle)

    for triangle in triangles:
        hull.append(triangle)

    # Update vectors of the new triangles
    def update_vectors():
        global new_triangle, opposite, triangle_point

        for current_triangle in new_triangles:
            opposite_vertex = None
            for vertex in [triangle.vertex1, triangle.vertex2, triangle.vertex3]:
                if vertex not in [current_triangle.vertex1, current_triangle.vertex2, current_triangle.vertex3]:
                    opposite_vertex = vertex
                    break

            if np.dot(current_triangle.vector, current_triangle.center - pts[opposite_vertex]) < 0:
                current_triangle.vector *= -1

    def points_in_pos_plane():
        update_triangle_pos_plane(new_triangles, pts)

    def shell():
        # Create edge list from the convex hull
        hull_edges = []
        for tri_mesh in hull:
            hull_edges.append([tri_mesh.vertex1, tri_mesh.vertex2])
            hull_edges.append([tri_mesh.vertex1, tri_mesh.vertex3])
            hull_edges.append([tri_mesh.vertex2, tri_mesh.vertex3])

        plot_fig, plot_ax = plt.subplots(subplot_kw={'projection': '3d'})

        # Plot points in 3D space
        plot_ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], color='red')

        # Plot edges of the convex hull
        for edge in hull_edges:
            plot_ax.plot3D([pts[edge[0]][0], pts[edge[1]][0]],
                           [pts[edge[0]][1], pts[edge[1]][1]],
                           [pts[edge[0]][2], pts[edge[1]][2]],
                           color='blue')

        plt.show()

    counter = 0
    while len(list_of_triangles) > 0:
        # Remove the same triangles
        hull = list(set(hull))

        if interactive:
            shell()
        counter = counter + 1
        if counter > N ** 2:
            break

        # Take a triangle from the stack
        triangle = None
        # Triangles without positive points are on the hull
        for stack_triangle in list_of_triangles:
            if len(stack_triangle.pos_plane) == 0:
                triangle = stack_triangle
                break
        if triangle is None:
            triangle = list_of_triangles.pop()
        else:
            list_of_triangles.remove(triangle)
            if triangle in hull:
                hull.remove(triangle)

        if len(triangle.pos_plane) == 0:
            hull.append(triangle)
            continue

        # Find the point with the maximum distance from the plane (for triangles not on the hull)
        max_point = max(triangle.pos_plane, key=lambda p: p[1])

        # Find the triangles with the point in the positive plane
        positive_triangles = [stack_triangle for stack_triangle in list_of_triangles
                              if np.dot(stack_triangle.vector, pts[max_point[0]] - stack_triangle.center) > 0]

        if len(positive_triangles) == 0:
            new_triangles = [Triangle(triangle.vertex1, triangle.vertex2, max_point[0]),
                             Triangle(triangle.vertex1, triangle.vertex3, max_point[0]),
                             Triangle(triangle.vertex2, triangle.vertex3, max_point[0])]

            # Update vectors of the new triangles
            update_vectors()
            points_in_pos_plane()

            # Add new triangles to the list (hull) + final triangle for a complete hull
            for new_triangle in new_triangles:
                list_of_triangles.append(new_triangle)
                hull.append(new_triangle)
            if triangle in hull:
                hull.remove(triangle)

        else:
            positive_triangles.append(triangle)
            triangle_dict = {}

            # Adding vertices to the dictionary, then removing triangles from the hull for correct updating
            for positive_triangle in positive_triangles:
                for point in [positive_triangle.vertex1, positive_triangle.vertex2, positive_triangle.vertex3]:
                    triangle_dict[point] = triangle_dict.get(point, 0) + 1

            for positive_triangle in positive_triangles:
                if positive_triangle in hull:
                    hull.remove(positive_triangle)

            # New triangles made from hull's edges and points in the group
            group = list(triangle_dict.keys())
            edges = []
            for hull_triangle in hull:
                for edge in [[hull_triangle.vertex1, hull_triangle.vertex2],
                             [hull_triangle.vertex1, hull_triangle.vertex3],
                             [hull_triangle.vertex2, hull_triangle.vertex3]]:
                    if edge[0] in group and edge[1] in group:
                        edges.append(edge)

            # Remove the same edges
            edges_to_remove = []
            edge_dict = {}
            for edge in edges:
                # Unique key for the edge, same ones are added to the list
                edge_key = str(min(edge[0], edge[1])) + "-" + str(max(edge[0], edge[1]))
                if edge_key in edge_dict:
                    edges_to_remove.append(edge)
                else:
                    edge_dict[edge_key] = 1

            # Remove the edges that appear multiple times
            for edge_to_remove in edges_to_remove:
                edges = [edge for edge in edges if edge != edge_to_remove]

            # Create new triangles from the edges left in the list
            new_triangles = [Triangle(edge[0], edge[1], max_point[0]) for edge in edges]
            update_vectors()
            points_in_pos_plane()

            for new_triangle in new_triangles:
                list_of_triangles.append(new_triangle)
                hull.append(new_triangle)

            list_of_triangles = [t for t in list_of_triangles if t not in positive_triangles]

    # Remove doubled triangles
    hull = list(set(hull))

    if display_hull:
        shell()

    return hull


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Generate a convex hull.')
    arg_parser.add_argument('--N', type=int, default=10)
    arg_parser.add_argument('--interactive', action='store_true', default=False)
    arg_parser.add_argument('--display_hull', action='store_true', default=False)
    arg_parser.add_argument('--display_tetrahedron', action='store_true', default=False)
    args = arg_parser.parse_args()

    N = args.N
    interactive = args.interactive
    display_hull = args.display_hull
    display_tetrahedron = args.display_tetrahedron

    if interactive or display_hull or display_tetrahedron:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/path/to/Qt/plugins/platforms'
        matplotlib.use('Qt5Agg')

    pts = np.random.normal(loc=0, scale=2.0, size=(N, 3))
    hull = quickhull()
    
