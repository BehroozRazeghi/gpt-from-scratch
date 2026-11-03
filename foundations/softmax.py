import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # pass

       shiftZ = z - np.max(z)
       expsZ = np.exp(shiftZ)
       res = np.round(expsZ / np.sum(expsZ), 4)
       return res

