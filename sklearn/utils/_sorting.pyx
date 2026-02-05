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
    
    Fixes #33167: replaces recursive quicksort with iterative version to prevent
    stack overflow with datasets containing many duplicate values.
    
    The numpy equivalent is:
        def simultaneous_sort(dist, idx):
             i = np.argsort(dist)
             return dist[i], idx[i]
    """
    cdef:
        intp_t i, j, low, high, pivot_idx, store_idx
        floating pivot_val
        # Fixed stack: stores [low0, high0, low1, high1, ...]
        intp_t stack[256]  # 128 pairs = safe for 1.6M+
        intp_t top = 0
    
    # Handle small arrays efficiently (unchanged)
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
    
    # Push initial range [0, size-1]
    stack[top] = 0
    top += 1
    stack[top] = size - 1
    top += 1
    
    while top > 0:
        # Pop range from stack
        top -= 1
        high = stack[top]
        top -= 1
        low = stack[top]
        
        # Insertion sort for small ranges (< 16 elements)
        if high - low < 16:
            for i in range(low + 1, high + 1):
                pivot_val = values[i]
                j = i - 1
                while j >= low and values[j] > pivot_val:
                    values[j + 1] = values[j]
                    indices[j + 1] = indices[j]
                    j -= 1
                values[j + 1] = pivot_val
                indices[j + 1] = indices[i]
            continue
        
        # Median-of-three pivot selection (unchanged)
        pivot_idx = low + (high - low) // 2
        if values[low] > values[high]:
            dual_swap(values, indices, low, high)
        if values[high] > values[pivot_idx]:
            dual_swap(values, indices, high, pivot_idx)
            if values[low] > values[high]:
                dual_swap(values, indices, low, high)
        
        pivot_val = values[high]
        
        # Partition (unchanged)
        store_idx = low
        for i in range(low, high):
            if values[i] < pivot_val:
                if i != store_idx:
                    dual_swap(values, indices, i, store_idx)
                store_idx += 1
        
        if store_idx != high:
            dual_swap(values, indices, store_idx, high)
        
        pivot_idx = store_idx
        
        # Push partitions to stack (larger first = optimal stack usage)
        cdef intp_t left_size = pivot_idx - low
        cdef intp_t right_size = high - pivot_idx
        
        if left_size > right_size:
            # Left is larger: push left first
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
            # Right is larger/equal: push right first
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
