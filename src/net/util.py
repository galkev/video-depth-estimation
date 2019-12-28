

def same_padding(kernel_size, size=0, stride=1, dilation=1):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
