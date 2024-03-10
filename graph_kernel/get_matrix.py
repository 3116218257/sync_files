import torch

def get_A(N):
    row_vector = torch.arange(N).unsqueeze(0)
    column_vector = torch.arange(N).unsqueeze(1)
    exponent = - 2 * torch.abs(row_vector - column_vector)
    A = 2.0 ** exponent

    return A

def get_D(A):
    D = torch.zeros(A.shape[0], A.shape[0])
    D.diagonal().copy_(A.diagonal())
    return D

def get_L_normalized(A, D):
    L = torch.eye(A.shape[0])
    diag_D = torch.zeros(A.shape[0], A.shape[0])
    diag_D.diagonal().copy_(D.diagonal())
    diag_D = torch.inverse(diag_D) ** (0.5)
    # print(diag_D @ A @ diag_D, "\n", A)
    return L - diag_D @ A @ diag_D, diag_D @ A @ diag_D
#

def expand_matrix(X, batch_size):
    batch_tensor = X.unsqueeze(0)
    batch_tensor = batch_tensor.expand(batch_size, -1, -1)
    return batch_tensor


if __name__ == '__main__':
    A = get_A(32)
    D = get_D(A)
    L, K = get_L_normalized(A, D)
    print(K)

    A = expand_matrix(A, 4)
    D = expand_matrix(D, 4)
    L = expand_matrix(L, 4)
    K = expand_matrix(K, 4)