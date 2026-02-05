from cython cimport floating
from libc.stdint cimport intp_t

cdef inline void dual_swap(
    floating* darr,
    intp_t *iarr,
    intp_t a,
    intp_t b,
) noexcept nogil:
    """Swap the values at index a and b of both darr and iarr"""
    cdef floating dtmp = darr[a]
    darr[a] = darr[b]
    darr[b] = dtmp

    cdef intp_t itmp = iarr[a]
    iarr[a] = iarr[b]
    iarr[b] = itmp


cdef int simultaneous_sort(
    floating* values,
    intp_t* indices,
    intp_t size,
) noexcept nogil:
    """
    Perform an iterative quicksort on the values array as to sort them ascendingly.
    This simultaneously performs the swaps on both the values and the indices arrays.

    The numpy equivalent is:

        def simultaneous_sort(dist, idx):
             i = np.argsort(dist)
             return dist[i], idx[i]

    Notes
    -----
    Arrays are manipulated via a pointer to there first element and their size
    as to ease the processing of dynamically allocated buffers.
    """
    # TODO: In order to support discrete distance metrics, we need to have a
    # simultaneous sort which breaks ties on indices when distances are identical.
    # The best might be using a std::stable_sort and a Comparator which might need
    # an Array of Structures (AoS) instead of the Structure of Arrays (SoA)
    # currently used.
    cdef:
        intp_t i, j, low, high, pivot_idx, store_idx
        floating pivot_val
        intp_t pivot_idx_val
        intp_t stack[256]
        intp_t top = 0

    if size <= 1:
        return 0
    elif size == 2:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
        return 0
    elif size == 3:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
        if values[1] > values[2]:
            dual_swap(values, indices, 1, 2)
            if values[0] > values[1]:
                dual_swap(values, indices, 0, 1)
        return 0
    
    stack[top] = 0
    top += 1
    stack[top] = size - 1
    top += 1
    
    while top > 0:
        top -= 1
        high = stack[top]
        top -= 1
        low = stack[top]
        
        if high - low < 16:
            for i in range(low + 1, high + 1):
                pivot_val = values[i]
                pivot_idx_val = indices[i]
                j = i - 1
                while j >= low and values[j] > pivot_val:
                    values[j + 1] = values[j]
                    indices[j + 1] = indices[j]
                    j -= 1
                values[j + 1] = pivot_val
                indices[j + 1] = pivot_idx_val
            continue
        
        pivot_idx = low + (high - low) // 2
        if values[low] > values[high]:
            dual_swap(values, indices, low, high)
        if values[high] > values[pivot_idx]:
            dual_swap(values, indices, high, pivot_idx)
            if values[low] > values[high]:
                dual_swap(values, indices, low, high)
        
        pivot_val = values[high]
        
        store_idx = low
        for i in range(low, high):
            if values[i] < pivot_val:
                if i != store_idx:
                    dual_swap(values, indices, i, store_idx)
                store_idx += 1
        
        if store_idx != high:
            dual_swap(values, indices, store_idx, high)
        
        pivot_idx = store_idx
        
        cdef intp_t left_size = pivot_idx - low
        cdef intp_t right_size = high - pivot_idx
        
        if left_size > right_size:
            if left_size > 3:
                stack[top] = low
                top += 1
                stack[top] = pivot_idx - 1
                top += 1
            if right_size > 3:
                stack[top] = pivot_idx + 1
                top += 1
                stack[top] = high
                top += 1
        else:
            if right_size > 3:
                stack[top] = pivot_idx + 1
                top += 1
                stack[top] = high
                top += 1
            if left_size > 3:
                stack[top] = low
                top += 1
                stack[top] = pivot_idx - 1
                top += 1
    
    return 0
