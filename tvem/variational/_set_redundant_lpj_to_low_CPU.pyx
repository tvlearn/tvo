# cython: boundscheck=False
# cython: language_level=3
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
    cdef int n, s, ss, next_s
    for n in range(N): # for each datapoint
        for s in range(Snew): # for each new state
            next_s = s + 1
            for ss in range(next_s, Snew): # check if equal to other new states
                if is_equal(new_states[n, s], new_states[n, ss]):
                    new_lpj[n, s] = low_lpj
                    break
            else: # did not find a duplicate in new states, so search old states too
                for ss in range(S):
                    if is_equal(new_states[n, s], old_states[n, ss]):
                        new_lpj[n, s] = low_lpj
                        break
