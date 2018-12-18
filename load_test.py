

from ruamel.yaml import YAML
import sys
import numpy as np
import torch
import os


def test_slice(arr, rows, cols):
    new_arr = np.zeros((rows, cols), dtype=np.float32)
    for i in range(int(len(arr) / cols)):
        pos = i * cols
        val = arr[pos:pos + cols]
        new_arr[i] = val
    return new_arr

def load_data(path, lable):
    f = open(path, "r")
    yaml = YAML(typ='safe')
    data = yaml.load(f)
    training = data['Training']
    rows = training['rows']
    cols = training['cols']
    arr = training['data']
    result_arr = test_slice(arr, rows, cols)
    lables_arr = np.full((rows, 1), lable, dtype=np.float32)
    return result_arr, lables_arr, rows, cols


def load_positive_data(path_positive):
    return load_data(path_positive, 1.0)

def load_negative_data(path_negative):
    return load_data(path_negative, -1.0)


def load_training_data(input_path, res_path):
    inp_exists = os.path.isfile(input_path)
    res_exists = os.path.isfile(res_path)

    print("Load training data")
    if (inp_exists and res_exists):
        x = torch.load(input_path)
        y = torch.load(res_path)

        pass
    else :
        path_positive = ''
        path_positive = "/home/mike/workspace/tmp/opencv_patterns_compare/build/mikes_training_data.mat"
        positive_data, positive_lables, pos_rows, pos_cols = load_positive_data(path_positive)
        print("positive data loaded")
        print(torch.from_numpy(positive_data).size())
        print(torch.from_numpy(positive_lables).size())

        path_negative = "/home/mike/workspace/tmp/opencv_patterns_compare/build/ilya_training_data.mat"
        negative_data, negative_lables, neg_rows, neg_cols = load_negative_data(path_negative)
        print("negative data loaded")
        print(torch.from_numpy(negative_data).size())
        print(torch.from_numpy(negative_lables).size())

        x = np.vstack([positive_data, negative_data])
        y = np.vstack([positive_lables, negative_lables])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        torch.save(x, input_path)
        torch.save(y, res_path)

        pass
    return x, y

def load_sample_data(data_path):
    test_samples_data_path = "test_samples.dat"

    test_samples_exists = os.path.isfile(data_path)

    if test_samples_exists:
        test_samples = torch.load(data_path)
        pass
    else:
        sample_0_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_0.png.mat"
        sample_1_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_1.png.mat"
        sample_2_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_2.png.mat"
        sample_3_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_3.png.mat"
        sample_4_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_4.png.mat"
        sample_5_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_5.png.mat"
        sample_6_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/mike_test_series/photo_6.png.mat"

        def add_test_sample(arr, path):
            data, lables, neg_rows, neg_cols = load_positive_data(path)
            new_arr = np.vstack([arr, data])
            print("Loaded test sample: " + path)
            return new_arr

        print("Load test samples")
        test_samples, lab, rp, cp = load_data(sample_0_path, 1)
        test_samples = add_test_sample(test_samples, sample_1_path)
        test_samples = add_test_sample(test_samples, sample_2_path)
        test_samples = add_test_sample(test_samples, sample_3_path)
        test_samples = add_test_sample(test_samples, sample_4_path)
        test_samples = add_test_sample(test_samples, sample_5_path)
        test_samples = add_test_sample(test_samples, sample_6_path)

        for x in neg_samples:
            print(x)
            test_samples = add_test_sample(test_samples, x)
            pass

        test_samples = torch.from_numpy(test_samples)
        torch.save(test_samples, data_path)
        pass
    return test_samples


# =============================  data

neg_samples = [
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2205.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2247.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2260.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2332.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2365.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/1858.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2208.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2249.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2261.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2334.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/1859.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2210.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2250.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2263.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2337.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2352.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/1862.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2238.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2251.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2264.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2341.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2356.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/1863.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2240.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2252.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2342.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2357.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2200.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2242.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2253.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2266.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2343.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2358.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2202.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2244.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2254.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2328.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2346.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2362.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2204.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2245.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2257.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2330.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/2348.jpg.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_3.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_4.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_0.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_7.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_10.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_8.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_11.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_9.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_1.png.mat",
    "/home/mike/workspace/tmp/opencv_patterns_compare/resources/test_series_1/photo_2.png.mat"
]