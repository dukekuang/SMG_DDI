import torch

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1-x2)**2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = torch.mean(sx1**k)
    ss2 = torch.mean(sx2**k)
    return l2diff(ss1, ss2)

class CMD(object):
    def __init__(self, n_moments=5):
        self.n_moments = n_moments

    def __call__(self, x1, x2):
        mx1 = torch.mean(x1, dim=0)
        mx2 = torch.mean(x2, dim=0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = l2diff(mx1, mx2)
        scms = dm

        for i in range(self.n_moments-1):
            # moment diff of centralized samples
            scms = scms + moment_diff(sx1, sx2, i+2) 
        return scms
