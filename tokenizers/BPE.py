from sortedcontainers import SortedSet

class BPETokenizer():
    def __init__(self, train_bytes):
        assert isinstance(train_bytes, bytes)
        self.priority = SortedSet()
        self.counts = {}
        self.nodes = {}
        self.last_idx = 255
        self.idx_map = {}

        self.start_node = {'idx': None, 'prev': None, 'next': None}
        self.end_node = {'idx': None, 'prev': None, 'next': None}

        prev_node = self.start_node
        for b in train_bytes:
            new_node = {'idx': b, 'prev': prev_node, 'next': None}
            prev_node['next'] = new_node
            self.incr(prev_node, new_node)
            prev_node = new_node
        prev_node['next'] = self.end_node

    def idx_str(self, idx):
        return self.idx_map.get(idx, chr(idx))

    def tokens(self):
        acc = []
        cur_node = self.start_node['next']
        while id(cur_node) != id(self.end_node):
            acc.append((cur_node['idx'], self.idx_str(cur_node['idx'])))
            cur_node = cur_node['next']
        return acc

    def get_new_idx(self, idx0, idx1):
        self.last_idx += 1
        self.idx_map[self.last_idx] = self.idx_str(idx0) + self.idx_str(idx1)
        return self.last_idx

    def decr(self, node0, node1):
        idx0, idx1 = node0['idx'], node1['idx']
        if idx0 is None or idx1 is None:
            return
        nodemap = self.nodes[(idx0, idx1)]
        assert (id(node0), id(node1)) in nodemap
        del nodemap[(id(node0), id(node1))]
        if len(nodemap) == 0:
            del self.nodes[(idx0, idx1)]

        old_cnt = self.counts[(idx0, idx1)]
        new_cnt = old_cnt - 1
        self.priority.remove((old_cnt, idx0, idx1))
        if new_cnt > 0:
            self.priority.add((new_cnt, idx0, idx1))
            self.counts[(idx0, idx1)] = new_cnt
        else:
            del self.counts[(idx0, idx1)]

    def incr(self, node0, node1):
        idx0, idx1 = node0['idx'], node1['idx']
        if idx0 is None or idx1 is None:
            return
        nodemap = self.nodes.setdefault((idx0, idx1), {})
        nodemap[(id(node0), id(node1))] = (node0, node1)

        old_cnt = self.counts.setdefault((idx0, idx1), 0)
        new_cnt = old_cnt + 1
        if old_cnt > 0:
            self.priority.remove((old_cnt, idx0, idx1))

        self.priority.add((new_cnt, idx0, idx1))
        self.counts[(idx0, idx1)] = new_cnt

    def nodes_for(self, idx0, idx1):
        while True:
            if (idx0, idx1) in self.nodes:
                yield next(iter(self.nodes[(idx0, idx1)].values()))
            else:
                return

    def step(self):
        _, idx0, idx1 = self.priority[-1]
        new_idx = self.get_new_idx(idx0, idx1)
        for node0, node1 in self.nodes_for(idx0, idx1):
            self.decr(node0, node1)
            assert node0['next'] == node1
            prev_node = node0['prev']
            self.decr(prev_node, node0)
            next_node = node1['next']
            self.decr(node1, next_node)

            new_node = {'idx': new_idx, 'prev': node0['prev'], 'next': node1['next']}

            prev_node['next'] = new_node
            self.incr(prev_node, new_node)
            next_node['prev'] = new_node
            self.incr(new_node, next_node)

