import torch
from torch import cat, stack
from torch import maximum, tensor
from torch import Tensor, einsum

def balanced_mcr2_loss_wrapper(Z, labels, eps=0.1, num_samples_class0=1000):
    """
    Tính MCR2 loss nhưng giảm thiểu sự lấn át của Class 0.

    Args:
        Z: (Batch, N, D) hoặc (N, D) - Features
        labels: (Batch, N) hoặc (N,) - Labels (Integer, không phải one-hot)
        eps: Hệ số epsilon của MCR2
        num_samples_class0: Số lượng điểm tối đa của Class 0 muốn giữ lại (mặc định 1000).
                            Các class khác được giữ nguyên 100%.
    """
    if Z.dim() == 3:
        Z_flat = Z.reshape(-1, Z.shape[-1])
        labels_flat = labels.reshape(-1)
    else:
        Z_flat = Z
        labels_flat = labels
    idx_class0 = (labels_flat == 0).nonzero(as_tuple=True)[0]
    idx_others = (labels_flat != 0).nonzero(as_tuple=True)[0]
    if idx_class0.size(0) > num_samples_class0:
        perm = torch.randperm(idx_class0.size(0), device=Z.device)
        idx_class0_keep = idx_class0[perm[:num_samples_class0]]
    else:
        idx_class0_keep = idx_class0
    indices_to_use = torch.cat([idx_class0_keep, idx_others])
    Z_balanced = Z_flat[indices_to_use]
    labels_balanced = labels_flat[indices_to_use]
    num_classes = 25
    y_onehot = torch.nn.functional.one_hot(labels_balanced.long(), num_classes=num_classes)
    loss = supervised_mcr2_loss(Z_balanced.unsqueeze(0), y_onehot.unsqueeze(0), eps)

    return loss

def tensorized_ZtZ(Z: Tensor):
    """
    For a batch of matrices Z, computes a vectorized Z.T @ Z.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices.

    Returns:
        Tensor of shape (*, D, D); the matrices Z.T @ Z.
    """
    ZtZ = einsum("...ni, ...nj -> ...ij", Z, Z)  # (*, D, D)
    return ZtZ  # (*, D, D)


def tensorized_ZtZ_class(Z: Tensor, y_onehot: Tensor):
    """
    For a batch of matrices Z, computes a vectorized Zi.T @ Zi for each class i.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.

    Returns:
        Tensor of shape (*, K, D, D); the matrices Zi.T @ Zi.
    """
    ZtZ = einsum("...ni, ...nj, ...nk -> ...kij", Z, Z, y_onehot.float())  # (*, K, D, D)
    return ZtZ  # (*, K, D, D)


def second_moment(Z: Tensor):
    """
    Computes the empirical second moment of the input matrices.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.

    Returns:
        Tensor of shape (*, D, D); the empirical second moment of Z.
    """
    N = tensor(Z.shape[-2], device=Z.device)  # ()
    N = maximum(N, tensor(1.0, device=Z.device))  # (*)
    ZtZ = tensorized_ZtZ(Z)  # (*, D, D)
    return ZtZ / N  # (*, D, D)


def second_moment_class(Z: Tensor, y_onehot: Tensor):
    """
    Computes the empirical second moment of each class of the input matrices.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.

    Returns:
        Tensor of shape (*, K, D, D); the empirical second moment of each class of Z.
    """
    ZtZ = tensorized_ZtZ_class(Z, y_onehot)  # (*, K, D, D)
    Nc = y_onehot.float().sum(dim=-2).unsqueeze(-1).unsqueeze(-1)  # (*, K, 1, 1)
    Nc = maximum(Nc, tensor(1.0, device=ZtZ.device))  # (*, K, 1, 1)
    return ZtZ / Nc  # (*, K, D, D)


def gramian(Z: Tensor):
    """
    Computes the Gramian of the rows of each input matrix.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.

    Returns:
        Tensor of shape (*, N, N); the Gramian of the rows of Z.
    """
    ZZt = einsum("...ni, ...mi -> ...mn", Z, Z)  # (*, N, N)
    return ZZt  # (*, N, N)


def logdet_I_plus(M: torch.Tensor):
    """
    Computes log det(I + M) for a Hermitian PSD matrix M.
    Supports batched input M: (..., D, D)
    """

    # Ensure symmetry (fix float numerical drift)
    M = (M + M.transpose(-1, -2)) * 0.5

    # Add jitter to stabilize eigen decomposition
    jitter = 1e-4  # nếu vẫn lỗi tăng 1e-3
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    M = M + jitter * I

    # Stable eigenvalue computation
    ev = torch.linalg.eigvalsh(M)

    # Avoid negative eigenvalues due to floating precision
    ev = torch.clamp_min(ev, 0.0)

    # log det(I + M) = sum(log(1 + λ_i))
    return torch.log1p(ev).sum(dim=-1)

def R(Z: Tensor, eps: float):
    """
    Computes the "coding rate" of the input matrix with respect to the given quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate of Z.
    """
    N, D = Z.shape

    P = second_moment(Z)  # (D, D)

    ld = logdet_I_plus(D / (eps ** 2) * P)  # ()

    return 0.5 * ld  # ()


def Rc(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the "segmented coding rate" of the input matrix with respect to the given class assignment and quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian conditional on their class.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (N, K); a {0, 1} matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the segmented coding rate of Z with respect to y.
    """
    N, D = Z.shape

    Nc = y_onehot.float().sum(dim=0)  # (K, )
    pi = Nc / N  # (K, )

    P = second_moment_class(Z, y_onehot)  # (K, D, D)

    ld = logdet_I_plus(D / (eps ** 2) * P)  # (K, )
    return 0.5 * (pi * ld).sum()  # ()


def DeltaR(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the "coding rate reduction" of the input matrix with respect to the given class assignment and quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian conditional on their class.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (*); the coding rate reduction of Z with respect to y.
    """
    *batch_dims, N, D = Z.shape
    N_t = torch.tensor(float(N), device=Z.device, dtype=Z.dtype)

    Nc = y_onehot.float().sum(dim=-2)  # (*, K)
    pi = Nc / N_t  # (*, K)

    P = second_moment_class(Z, y_onehot)  # (*, K, D, D)
    # P_com = sum_k pi_k * P_k   (*, D, D)
    pi_exp = pi.unsqueeze(-1).unsqueeze(-1)  # (*, K, 1, 1)
    P_com = torch.sum(pi_exp * P, dim=-3)  # (*, D, D)
    # P_tot = cat(P, P_com) along K dim (dim=-3)
    P_com_for_cat = P_com.unsqueeze(-3)  # (*, 1, D, D)
    P_tot = torch.cat((P, P_com_for_cat), dim=-3)  # (*, K+1, D, D)

    scale = D / (eps ** 2)
    ld = logdet_I_plus(scale * P_tot)  # (*, K+1)

    delta_r = 0.5 * (ld[..., -1] - torch.sum(pi * ld[..., :-1], dim=-1))  # (*)
    return delta_r


def DeltaR_diff(Z1: Tensor, Z2: Tensor, eps: float):
    """
    Computes the "coding rate reduction" difference between the two input matrices with respect to the given quantization error, assuming that the rows of the input matrices are distributed according to a zero-mean Gaussian.

    Args:
        Z1: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        Z2: Tensor of shape (M, D); a matrix where each row is a data point and each column is a feature.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate reduction difference between Z and Zhat.
    """
    N, D = Z1.shape
    M, D = Z2.shape

    N = tensor(float(N), device=Z1.device).unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    M = tensor(float(M), device=Z1.device).unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    T = maximum(M + N, tensor(1.0, device=Z1.device))  # (1, 1)

    P_Z1 = second_moment(Z1)  # (D, D)
    P_Z2 = second_moment(Z2)  # (D, D)
    P_com = (N * P_Z1 + M * P_Z2) / T  # (D, D)
    P_tot = stack((P_Z1, P_Z2, P_com), dim=0)  # (3, D, D)

    ld = logdet_I_plus(D / (eps ** 2) * P_tot)  # (3, )

    N = N.squeeze(-1).squeeze(-1)  # ()
    M = M.squeeze(-1).squeeze(-1)  # ()
    T = T.squeeze(-1).squeeze(-1)  # ()

    return 0.5 * (ld[2] - (N / T) * ld[0] - (M / T) * ld[1])  # ()


def DeltaR_cdiff(Z1: Tensor, Z2: Tensor, y1_onehot: Tensor, y2_onehot: Tensor, eps: float):
    """
    Computes the "coding rate reduction" difference between the two input matrices with respect to the given class assignments and quantization error, assuming that the rows of the input matrices are distributed according to a zero-mean Gaussian conditioned on class.

    Args:
        Z1: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        Z2: Tensor of shape (M, D); a matrix where each row is a data point and each column is a feature.
        y1_onehot: Tensor of shape (N, K); a matrix where each row is a data point and each column is a class.
        y2_onehot: Tensor of shape (M, K); a matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate reduction difference between Z and Zhat with respect to y1 and y2.
    """
    N, D = Z1.shape
    M, D = Z2.shape

    Nc = y1_onehot.float().sum(dim=0).unsqueeze(-1).unsqueeze(-1)  # (K, 1, 1)
    Mc = y2_onehot.float().sum(dim=0).unsqueeze(-1).unsqueeze(-1)  # (K, 1, 1)
    Tc = maximum(Nc + Mc, tensor(1.0, device=Z1.device))  # (K, 1, 1)

    P_Z1 = second_moment_class(Z1, y1_onehot)  # (K, D, D)
    P_Z2 = second_moment_class(Z2, y2_onehot)  # (K, D, D)
    P_com = (Nc * P_Z1 + Mc * P_Z2) / Tc  # (K, D, D)
    P_tot = stack((P_Z1, P_Z2, P_com), dim=0)  # (3, K, D, D)

    ld = logdet_I_plus(D / (eps ** 2) * P_tot)  # (3, K)

    Nc = Nc.squeeze(-1).squeeze(-1)  # (K, )
    Mc = Mc.squeeze(-1).squeeze(-1)  # (K, )
    Tc = Tc.squeeze(-1).squeeze(-1)  # (K, )

    return 0.5 * (ld[2] - (Nc / Tc) * ld[0] - (Mc / Tc) * ld[1]).sum()  # ()

def supervised_mcr2_loss(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the MCR2 loss from "Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction" by Yu et al.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the MCR2 loss of Z with respect to y (averaged over batch dims).
    """
    delta_r = DeltaR(Z, y_onehot, eps)
    return delta_r.mean()