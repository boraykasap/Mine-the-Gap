import random
from typing import List

class NegativeSampler:
    """Sampler semplice, uniforme sui nodi. K negativi per positivo."""
    def __init__(self, num_nodes: int, k: int = 5, seed: int = 42):
        self.num_nodes = num_nodes
        self.k = k
        self.rng = random.Random(seed)

    def sample_for_src_batch(self, src_ids) -> List[int]:
        """
        Genera K negativi per ciascun src del batch, restituendo
        una lista flatten di dst negativi (len = B*K).
        """
        neg_dst = []
        for s in src_ids:
            for _ in range(self.k):
                d = self.rng.randrange(0, self.num_nodes)
                while d == int(s):  # evita self-loop banale
                    d = self.rng.randrange(0, self.num_nodes)
                neg_dst.append(d)
        return neg_dst
