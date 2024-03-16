import torch
import numpy as np
def get_K(ind1, ind2, coeff):
    row_vector = ind1.unsqueeze(1)
    column_vector = ind2.unsqueeze(2)
    # print(row_vector, '\n', column_vector)
    exponent = coeff * torch.exp(torch.abs(row_vector - column_vector))
    # exponent = coeff * torch.abs(row_vector - column_vector)
    K = exponent ** (-1)
    
    return K

def recover_k(psi1, psi2, eigen_value,):
    '''
    psi: [frames, k]
    eigen_value: [k, 1]
    
    '''
    eigen_value = eigen_value.view(eigen_value.shape[0], 1)
    k_ij = (eigen_value * psi1.T * psi2.T).sum(dim=0)

    return k_ij

def difference_recovered_true(k_true, k_ij_recover, i, j):
    return (k_true[i, j] - k_ij_recover) ** 2


if __name__ == '__main__':

    pass
    # p = torch.randn(1, 10)
    # A = get_A(p, p)
    # print(A.shape)
    # ev = torch.randn(128, 1)
    # print(recover_k(p, p, ev).shape)
    # A = get_K(torch.Tensor(np.array([[1, 2, 3, 4]])), torch.Tensor(np.array([[1, 5, 3, 4]])), 1)
    # print(A)
    # D = get_D(A)
    # L, K = get_L_normalized(A, D)
    # print(K)

    # A = expand_matrix(A, 4)
    # D = expand_matrix(D, 4)
    # L = expand_matrix(L, 4)
    # K = expand_matrix(K, 4)

    # a = torch.Tensor([[1], [2], [3], [4]])
    # b = torch.ones(4,4)

    # print(a.shape, b.shape)
    # print((a * b).shape, (a * b).sum(0))  