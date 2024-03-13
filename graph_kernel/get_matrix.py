import torch
import numpy as np
def get_K(ind1, ind2):
    row_vector = ind1.unsqueeze(1)
    column_vector = ind2.unsqueeze(2)
    exponent = torch.exp(torch.abs(row_vector - column_vector))
    K = exponent ** (-1)
    
    return K

def recover_k(psi1, psi2, eigen_value):
    '''
    psi: [batch, k]
    eigen_value: [batch, k, k]
    
    '''
    eigen_value_list = eigen_value.diagonal(dim1=1,dim2=2)
    return (eigen_value_list * psi1 * psi2).sum(dim=1)

def difference_recovered_true(k_true, k_ij_recover, i, j):
    return (k_true[i, j] - k_ij_recover) ** 2

if __name__ == '__main__':
    # p = torch.randn(512, 10)
    # A = get_A(p, p)
    # print(A.shape)
    # ev = torch.randn(512, 128, 128)
    # print(recover_k(p, p, ev).shape)
    A = get_A(torch.Tensor(np.array([[1, 2, 3, 4], [1, 3, 3, 4]])), torch.Tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]])))
    print(A.shape)
    # D = get_D(A)
    # L, K = get_L_normalized(A, D)
    # print(K)

    # A = expand_matrix(A, 4)
    # D = expand_matrix(D, 4)
    # L = expand_matrix(L, 4)
    # K = expand_matrix(K, 4)