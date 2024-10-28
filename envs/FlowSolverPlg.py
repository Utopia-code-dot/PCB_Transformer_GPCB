import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
'0'  0  '3'
    ---   
 1 |   | 3
    ---
'1'  2  '2'
"""


# FlowSolver Used in PCBGPT
class FlowSolver:
    def __init__(self, tile: tuple = None, flows: list = None):
        self.tile = tile
        self.flows = flows
        self.lines = []

    def show_flows(self, mat=None, info=None):
        vertices, edges = self.gen_vertices(self.tile), self.gen_edges(self.tile)

        self.dis = [0] * 4
        for in_out in self.flows:
            for idx in range(2):
                if type(in_out[idx]) == int: self.dis[in_out[idx]] += 1

        all_cuts = [self.cut_edges(edges[i][0], edges[i][-1], self.dis[i]) for i in range(4)]

        self.flows = self.sort_flows(self.format_flows(self.flows))

        for flow in self.flows:
            order = self.get_sort_order(flow)[0]

            turn = None

            if order == -1:
                edge = int(flow[0])
                in_cut = vertices[edge]

                if flow[1] == self._next(edge, 0): out_cut = all_cuts[flow[1]].pop(-1)
                if flow[1] == self._next(edge, 1): out_cut = all_cuts[flow[1]].pop(0)

                x, y = vertices[edge]
                x_gap = int(0.25 * (self.tile[2] - self.tile[0]))
                y_gap = int(0.25 * (self.tile[3] - self.tile[1]))

                if edge == 0: turn = x + x_gap, y + y_gap
                if edge == 1: turn = x - x_gap, y + y_gap
                if edge == 2: turn = x - x_gap, y - y_gap
                if edge == 3: turn = x + x_gap, y - y_gap

            if order == 0:
                if flow[1] in ["e0", "e1"]:
                    in_cut = all_cuts[flow[0]].pop(0)

                if flow[1] == "e2":
                    in_cut = all_cuts[flow[0]].pop(-1)

                x_gap = int(0.25 * (self.tile[2] - self.tile[0]))
                y_gap = int(0.25 * (self.tile[3] - self.tile[1]))

                if flow[0] == 0: out_cut = in_cut[0] + x_gap, in_cut[1]
                if flow[0] == 1: out_cut = in_cut[0], in_cut[1] + y_gap
                if flow[0] == 2: out_cut = in_cut[0] - x_gap, in_cut[1]
                if flow[0] == 3: out_cut = in_cut[0], in_cut[1] - y_gap

            if order == 1:
                in_cut = all_cuts[flow[0]].pop(-1)
                out_cut = all_cuts[flow[1]].pop(0)

            if order == 2:
                in_cut = vertices[int(flow[0])]

                offset = (flow[1] + 4 - int(flow[0])) % 4
                if offset == 2: out_cut = all_cuts[flow[1]].pop(0)
                if offset == 3: out_cut = all_cuts[flow[1]].pop(-1)

            if order == 3:
                in_cut = all_cuts[flow[0]].pop(-1)
                out_cut = all_cuts[flow[1]].pop(0)

            if not turn:
                self.lines.append([in_cut, out_cut])
            else:
                self.lines.append([in_cut, turn, out_cut])

        # draw lines
        IsMat = True
        if not isinstance(mat, np.ndarray):
            IsMat = False
            mat = np.zeros((self.tile[2] + 1, self.tile[3] + 1), dtype=int)

        for line in self.lines:
            for i in range(0, len(line) - 1):
                cv2.line(mat, (line[i][1], line[i][0]), (line[i + 1][1], line[i + 1][0]), 255, 5)

        if not IsMat:
            if not info:
                plt.matshow(mat), plt.show()
            else:
                plt.matshow(mat), plt.savefig(info, dpi=500), plt.close()

    def _next(self, num, step):
        return (num + step) % 4

    def get_sort_order(self, flow):
        if type(flow[0]) == int and type(flow[1]) == str: return 0, flow[0], 0

        # (int, int)
        if type(flow[0]) == int and type(flow[1]) == int:

            # opposite
            if abs(flow[1] - flow[0]) == 2: return 3, flow[0], flow[1]

            # adjacent
            if abs(flow[1] - flow[0]) != 2: return 1, flow[0], flow[1]

        # (str, int)
        if type(flow[0]) == str and type(flow[1]) == int:

            edge = int(flow[0])

            # short
            if flow[1] in [self._next(edge, 0), self._next(edge, 1)]: return -1, int(flow[0]), flow[1]

            # long
            if flow[1] in [self._next(edge, 2), self._next(edge, 3)]: return 2, int(flow[0]), flow[1]

    def sort_flows(self, flows):
        return list(sorted(flows, key=self.get_sort_order))

    def format_flows(self, flows):
        new_flows = []
        for flow in flows:
            if flow in [(1, 0), (2, 1), (3, 2), (0, 3), (2, 0), (3, 1)]: flow = tuple(reversed(flow))
            new_flows.append(flow)
        return new_flows

    def gen_vertices(self, tile):
        return [(tile[0], tile[1]), (tile[2], tile[1]), (tile[2], tile[3]), (tile[0], tile[3])]

    def gen_edges(self, tile):
        edges = [[(tile[0], tile[1]), (tile[2], tile[1])], [(tile[2], tile[1]), (tile[2], tile[3])],
                 [(tile[2], tile[3]), (tile[0], tile[3])], [(tile[0], tile[3]), (tile[0], tile[1])]]
        edges.insert(0, edges.pop())
        return edges

    def cut_edges(self, p1, p2, n):
        d_x, d_y = p2[0] - p1[0], p2[1] - p1[1]
        cuts = [(int(p1[0] + d_x * i / (n + 1)), int(p1[1] + d_y * i / (n + 1))) for i in range(1, n + 1)]
        return cuts

    def get_cross(self, start1, end1, start2, end2):
        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = start1, end1, start2, end2
        det = lambda a, b, c, d: a * d - b * c
        d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
        p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
        q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
        if d != 0:
            lam, eta = p / d, q / d
            if not (0 <= lam <= 1 and 0 <= eta <= 1): return []
            return [lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2]
        if p != 0 or q != 0: return []
        t1, t2 = sorted([start1, end1]), sorted([start2, end2])
        if t1[1] < t2[0] or t2[1] < t1[0]: return []
        return max(t1[0], t2[0])


if __name__ == '__main__':
    tile_encode_mp = {1: [('1', 3), (1, 3)]}

    for k, v in tile_encode_mp.items():
        print(k, "\t:", v)
        FlowSolver(tile=(300, 700, 400, 800), flows=v).show_flows(k)
