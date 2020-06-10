import numpy as np


def im2col_dot(input_mat, kernel_mat, pad=2, stride=1):
    '''
    General matrix dot in convolution-wise, support pad and stride
    Calculate: kernel_mat * input_mat
    Args:
        input_mat: cin * col * hin * win, without pad
        kernel_mat: cout * cin * kh * kw
    Return:
        output_mat: cout * col * hout * wout
    '''
    input_mat = np.pad(input_mat, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    cin, col, H, W = input_mat.shape
    cout, _, kh, kw = kernel_mat.shape
    hout = (H - kh) // stride + 1
    wout = (W - kw) // stride + 1
    shapes = (cin, kh, kw, col, hout, wout)
    strides = input_mat.itemsize * np.array([H*W*col, W, 1, H*W, W*stride, stride])
    output = np.lib.stride_tricks.as_strided(input_mat, shape=shapes, strides=strides)
    output = output.reshape((cin*kh*kw, col*hout*wout))
    kernel_mat = kernel_mat.reshape(cout, -1)
    output_mat = np.dot(kernel_mat, output)  # cout * (col*hout*wout)
    return output_mat.reshape((cout, col, hout, wout))


def im2col_conv(input, w, b=None, pad=2, stride=1):
    '''
    input: shape = n * c_in * h_in * w_in  without pad
    w: weight, shape = c_out (#output channel) x c_in (#input channel) x kh (#kernel_size) x kw (#kernel_size)
    b: bias, shape = c_out
    Output:
        col_out: [n, c_out, h_out=h_in-k+1, w_out=w_in-k+1]
    '''
    N, C, H, W = input.shape
    cout, _, kh, kw = w.shape
    hout = (H - kh + 2 * pad) // stride + 1
    wout = (W - kw + 2 * pad) // stride + 1
    input = input.reshape(N, C, 1, H, W)
    #output = np.zeros((N, cout, hout, wout))
    #for i in range(N):
    #    output[i] = im2col_dot(input[i], w, pad, stride).reshape((cout, hout, wout))
    output = np.array([im2col_dot(input[i], w, pad, stride).reshape((cout, hout, wout)) for i in range(N)])
    if b is not None:
        output += b[np.newaxis, :, np.newaxis, np.newaxis]
    return output


def im2col(input, kh, kw, stride=1):
    '''
    input: N * C * H * W with pad
    out: N * (C*kh*kw) * (hout*wout)
    '''
    N, C, H, W = input.shape
    hout = (H - kh) // stride + 1
    wout = (W - kw) // stride + 1
    shapes = (N, C, kh, kw, hout, wout)
    strides = input.itemsize * np.array([C * H * W, H * W, W, 1, W * stride, stride])
    output = np.lib.stride_tricks.as_strided(input, shape=shapes, strides=strides)
    return output.reshape(N, C*kh*kw, hout*wout)


def conv2d_forward(input, w, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out(=h_in-k+1) x w_out(=w_in-k+1)
            where h_out, w_out is the height and width of output, after convolution
    '''
    return im2col_conv(input, w, b, pad)


def conv2d_backward(input, grad_output, w, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    # compute for grad_w and grad_b, no rotate!!!
    #input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    N, C, H, W = input.shape
    _, cout, hout, wout = grad_output.shape
    # grad_b = sum(grad_output), [n, c_out, h_out, w_out] -> [c_out]
    grad_b = grad_output.sum(axis=(0, 2, 3))# shape = c_out
    # grad_w = input * grad_output, [c_in, n, h_in, w_in] * [c_out, n, h_out, w_out] -> [c_out, c_in, k, k]
    # grad_w = np.zeros((cout, C, kernel_size, kernel_size))
    # for i in range(N):
    #     grad_w += im2col_dot(input[i].reshape((1, C, H, W)), grad_output[i].reshape((cout, 1, hout, wout)),  pad=pad)
    grad_w = np.sum([im2col_dot(input[i].reshape((1, C, H, W)), grad_output[i].reshape((cout, 1, hout, wout)),  pad=pad) for i in range(N)], axis=0)
    # compute for grad_input, attention for rotate 180!!!
    grad_out = np.pad(grad_output, ((0, 0), (0, 0), (kernel_size-1, kernel_size-1), (kernel_size-1, kernel_size-1)), 'constant')
    grad_w_loc = np.rot90(w.transpose((1, 0, 2, 3)), 2, axes=(2, 3))#rotate 180
    grad_input = im2col_conv(grad_out, grad_w_loc, pad=0)
    if pad != 0:
        grad_input = grad_input[:, :, pad:-pad, pad:-pad]
    return grad_input, grad_w, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    input = input.astype('float')
    if kernel_size == 2 and pad == 0:
        out = (input[..., ::2, ::2] + input[..., ::2, 1::2] + input[..., 1::2, ::2] + input[..., 1::2, 1::2]) * .25
        return out
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    N, C, H, W = input.shape
    hout = H // kernel_size
    wout = W // kernel_size
    output = im2col(input, kernel_size, kernel_size, stride=kernel_size).reshape([N, C, -1, hout, wout]).mean(axis=2)
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad = np.kron(grad_output, np.ones((kernel_size, kernel_size))) / (kernel_size * kernel_size)
    if pad == 0:
        return grad
    else:
        return grad[:, :, pad:-pad, pad:-pad]