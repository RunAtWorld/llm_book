import numpy as np
import random
import datetime




#从用例中获取tensor存储格式
def get_npu_format(type_format):
    type_dict = {
        "NCHW":0,"NHWC":1,"ND":2,"NC1HWC0":3,"FRACTAL_Z":4,"FRACTAL_NZ":29
    }
    return type_dict[type_format]

# 通过不同随机分布函数获取随机输入
def gen_data(data_shape, min_value, max_value, dtype, distribution='uniform'):
    try:
        # updated@2020/08/28 --
        if min_value == 'nan' or max_value == 'nan':
            return np.full(data_shape,np.nan).astype(dtype)
        elif min_value == 'inf' or max_value == 'inf':
            return np.full(data_shape, np.inf).astype(dtype)
        else:
            if 'uniform' in distribution:
                # 返回均匀分布随机值，min为随机最小值，max为随机最大值
                return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
            elif 'normal' in distribution:
                # 返回正态（高斯）分布随机值，min为正态分布中心值，max为正态分布标准差，需要大于0
                return np.random.normal(loc=min_value, scale=abs(max_value) + 1e-4, size=data_shape).astype(dtype)
            elif 'beta' in distribution:
                # 返回beta分布随机值，min为alpha，max为beta，两值均需要大于0
                return np.random.beta(a=abs(min_value) + 1e-4, b=abs(max_value) + 1e-4, size=data_shape).astype(dtype)
            elif 'laplace' in distribution:
                # 返回拉普拉斯分布随机值，min为拉普拉斯分布中心值，max为拉普拉斯分布指数衰减，需要大于0
                return np.random.laplace(loc=min_value, scale=abs(max_value) + 1e-4, size=data_shape).astype(dtype)
            elif 'triangular' in distribution:
                # 返回三角形分布随机值，min为三角形分布最小值，mode为三角形分布峰值，max三角形分布最大值
                mode = np.random.uniform(low=min_value, high=max_value)
                return np.random.triangular(left=min_value, mode=mode, right=max_value, size=data_shape).astype(dtype)
            elif 'relu' in distribution:
                # 返回均匀分布 + relu激活后的随机值
                data_pool = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
                return np.maximum(0, data_pool)
            elif 'sigmoid' in distribution:
                # 返回均匀分布 + sigmoid激活后的随机值
                data_pool = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
                return 1 / (1 + np.exp(-data_pool))
            elif 'softmax' in distribution:
                # 返回均匀分布 + softmax激活后的随机值
                data_pool = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
                return np.exp(data_pool) / np.sum(np.exp(data_pool))
            elif 'tanh' in distribution:
                # 返回均匀分布 + tanh激活后的随机值
                data_pool = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
                return (np.exp(data_pool) - np.exp(-data_pool)) / (np.exp(data_pool) + np.exp(-data_pool))
            elif 'arithmetic' in distribution:
                return np.linspace(min_value, max_value, data_shape[0])
            else:
                # 默认返回True, False的均匀分布，建议excecl设计表格range填写[0, 2]
                return np.random.uniform(low=0, high=2, size=data_shape).astype(np.int8).astype(dtype)
    except MemoryError:
        print_log('Shape %s is too large to invoke MemoryError' % data_shape, "ERROR")
# 保存数据到文件
def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == 'fp16':
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == 'fp32':
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == 'fp64':
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == 'int32':
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == 'int64':
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == 'uint8':
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == 'uint16':
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == 'uint32':
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == 'uint64':
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == 'complex64':
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == 'complex128':
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == 'bool':
        np.array(data_pool).astype(np.bool).tofile(data_path)

# 计算相对误差
def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

# Optimization t00505743,f00402938 @2020/06/10
def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

# 日志打印函数
def print_log(data=None, level='INFO'):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S"), level, data))


# 计算绝对误差
def cal_abs_diff(real_data, expect_data, diff_thd):
    return list(map(lambda x, y: cal_relative_diff(x, y, diff_thd), real_data, expect_data))

def display_output_tf(real_data, compare_data, expect_data, start, end, diff_thd):
    print_log('---------------------------------------------------------------------------------------')
    print_log('Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(NPU-FP16)  RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))

# 打印误差结果
def display_output(real_data, expect_data, start, end, diff_thd):
    print_log('---------------------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))

# 打印精度失败结果
def display_error_output_mp(real_data, expect_data, relative_diff, start, diff_thd):
    print_log('Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    count = 0
    for i in range(len(relative_diff)):
        j = i + start
        if relative_diff[j] > diff_thd:
            count += 1
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        if count == 100:
            break
    print_log('---------------------------------------------------------------------------------------')                

def display_error_output(real_data, expect_data, err_idx, relative_diff, start, end, diff_thd):
    print_log('Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if len_err <= 20 or count < 10 or count > len_err - 10:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10:
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s  \t %07s ' % (dot_3, dot_3, dot_3, dot_3, dot_3, dot_3))
    print_log('---------------------------------------------------------------------------------------')    

# 对比精度结果
def data_compare(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1):
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        print_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32),
                                 diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0
    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"
    print_log('---------------------------------------------------------------------------------------')
    print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print_log('---------------------------------------------------------------------------------------')
    print_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' % (diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Maximum error is: %s. Tolerance threshold is: %s.' % (max_error, max_diff_hd))
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff, start, end, diff_thd)
    return result, fulfill_percent, max_error    

# broadcast计算
# 减少维度
def broadcast_1(data_shape):
    broadcast_list = data_shape.copy()
    if len(broadcast_list) > 1:
        for i in range(0, random.randint(1, len(broadcast_list) - 1)):
            broadcast_list.pop(0)
    return broadcast_list

#维度不变
def broadcast_2(data_shape):
    broadcast_list = data_shape.copy()
    if len(broadcast_list) > 1:
        for i in range(0, random.randint(0, len(broadcast_list) - 1)):
            axis = random.randint(0, len(broadcast_list) - 1)
            broadcast_list[axis] = 1
    return broadcast_list

#增加维度
def broadcast_3(data_shape):
    broadcast_list = data_shape.copy()
    len_dims = len(data_shape)
    if len(data_shape) < 8:
        len_d = random.randint(len_dims + 1, 8)
        for i in range(0, random.randint(0, len(broadcast_list) - 1)):
            axis = random.randint(0, len(broadcast_list) - 1)
            broadcast_list[axis] = 1
        for i in range(len_d - len_dims):
            broadcast_list.insert(0, 1)
    return broadcast_list    