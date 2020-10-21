"""
create by: chenjianyi
create time: 2020.10.07 14:44
reference: https://github.com/willemmanuel/poisson-image-editing
           Patrick Perez, Michel Gangnet, and Andrew Blake's original paper on Poisson Image Editing
"""
from scipy.sparse import lil_matrix as lil_matrix
from scipy.sparse import linalg as linalg
import numpy as np

class PoissonImageEdit():
    def __init__(self):
        pass

    def mask_indices(self, mask):
        """Find the indicies where the mask is 1
        Params:
            mask: (h, w)
        Returns:
            indices: [(x1, y1), (x2, y2), ...], the indices of nonzero items in mask
        """
        nonzero = np.nonzero(mask)
        indices = list(zip(nonzero[0], nonzero[1]))
        return indices

    def get_surrounding(self, index):
        i, j = index
        return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

    def poisson_sparse_matrix(self, indices):
        N = len(indices)
        A = lil_matrix((N, N))  # initial is 0
        for i, index in enumerate(indices):
            A[i, i] = 4
            for x in self.get_surrounding(index):
                if x not in indices:
                    continue
                j = indices.index(x)
                A[i, j] = -1
        return A

    def laplacian_at_index(self, source, index):
        h, w, c = source.shape
        i, j = index
        value = 0
        if i + 1 < w:
            value += source[i, j] - source[i+1, j]
        if i - 1 >= 0:
            value += source[i, j] - source[i-1, j]
        if j + 1 < h:
            value += source[i, j] - source[i, j+1]
        if j - 1 >= 0:
            value += source[i, j] - source[i, j-1]
        #value = 4 * source[i, j] - source[i+1, j] - source[i-1, j] - source[i, j+1] - source[i, j-1]
        return value

    def in_omega(self, index, mask):
        return mask[index] == 1

    def edge(self, index, mask):
        if self.in_omega(index, mask) == False:
            return False
        for pt in self.get_surrounding(index):
            if self.in_omega(pt, mask) == False:
                return True
        return False

    def point_location(self, index, mask):
        if self.in_omega(index, mask):
            return 0  # inside
        if self.edge(index, mask) == True:
            return 1  # edge
        else:
            return 2  # outside

    def process(self, source, target, mask):
        indicies = self.mask_indices(mask)
        N = len(indicies)
        A = self.poisson_sparse_matrix(indicies)
        b = np.zeros((N, 3))
        for i, index in enumerate(indicies):
            #print(index, source.shape)
            b[i] = self.laplacian_at_index(source, index)
            if self.point_location(index, mask) == 1:
                for pt in self.get_surroending(index):
                    if self.in_omega(pt, mask) == False:
                        b[i] += target[pt]

        print(A.shape, b.shape)
        x = linalg.spsolve(A, b)
        print(x.shape)
        composite = np.copy(target).astype(int)
        for i, index in enumerate(indicies):
            composite[index] = x[i]
        return composite
