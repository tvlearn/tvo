import numpy as np
import cython

cdef bint is_equal(char[:] arr1, char[:] arr2) nogil:
    cdef size_t l = arr1.shape[0]
    cdef size_t i
    for i in range(l):
        if arr1[i] != arr2[i]:
            return False
    return True

cpdef void set_redundant_lpj_to_low_CPU(char[:, :, :] new_states, cython.floating[:, :] new_lpj, char[:, :, :] old_states) nogil:
    N = new_states.shape[0]
    Snew = new_states.shape[1]
    S = old_states.shape[1]
    low_lpj = -1e20
    for n in range(N): # for each datapoint
        for s in range(Snew): # for each new state
            for ss in range(Snew): # check if equal to other new states
                if s != ss and is_equal(new_states[n, s], new_states[n, ss]):
                    new_lpj[n, s] = low_lpj
                    break
            else: # check if equal to an old state
                for ss in range(S):
                    if is_equal(new_states[n, s], old_states[n, ss]):
                        new_lpj[n, s] = low_lpj
                        break
