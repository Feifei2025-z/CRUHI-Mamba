# Visualization of classification results
import numpy as np
import spectral as spy
import numpy as np


lab_colors_005 = np.array([
    [0, 0, 0],              # Background: Black
    [140, 208, 250],        # Class 1
    [254, 169, 0],          # Class 2
    [255, 144, 0],          # Class 3
    [253, 104, 77],         # Class 4
    [255, 195, 205],        # Class 5
    [240, 134, 134],        # Class 6
    [234, 154, 129],        # Class 7
    [1, 255, 255],          # Class 8
    [32, 148, 254],         # Class 9
    [106, 154, 238],        # Class 10
    [56, 206, 54],          # Class 11
    [0, 105, 0],            # Class 12
    [192, 185, 112],        # Class 13
    [255, 239, 213],        # Class 14
    [117, 135, 148],        # Class 15
    [127, 127, 127],        # Class 16
], dtype=np.uint8)


lab_colors_006 = np.array([
    [0, 0, 0],              # Background: Black
    [170, 46, 46],          # Class1
    [254, 169, 0],          # Class2
    [255, 144, 0],          # Class3
    [253, 104, 77],         # Class4
    [255, 195, 205],        # Class5
    [240, 134, 134],        # Class6
    [234, 154, 129],        # Class7
    [0, 208, 212],          # Class8
    [223, 24, 26],          # Class9
    [0, 105, 0],            # Class10
    [192, 185, 112],        # Class11
    [255, 239, 213],        # Class12
    [117, 135, 148],        # Class13
    [127, 127, 127],        # Class14
], dtype=np.uint8)


lab_colors_019 = np.array([
    [0, 0, 0],              # Background: Black
    [170, 46, 46],          # Class1
    [254, 169, 0],          # Class2
    [255, 144, 0],          # Class3
    [253, 104, 77],         # Class4
    [220, 168, 36],         # Class5
    [240, 134, 134],        # Class6
    [221, 117, 152],        # Class7
    [0, 208, 212],          # Class8
    [179, 238, 239],        # Class9
    [106, 154, 238],        # Class10
    [223, 24, 26],          # Class11
    [56, 206, 54],          # Class12
    [38, 143, 39],          # Class13
    [0, 105, 0],            # Class14
    [192, 185, 112],        # Class15
    [134, 0, 133],          # Class16
    [255, 239, 213],        # Class17
    [242, 242, 242],        # Class18
    [200, 205, 213],        # Class19
    [214, 214, 214],        # Class20
    [117, 135, 148],        # Class21
    [127, 127, 127],        # Class22
], dtype=np.uint8)


def visualize_predict(gt, predict_label, save_predict_path, save_gt_path, only_vis_label=False, colors=None):

    if colors is None:
        colors = lab_colors_005  # Default color mapping

    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label, (row, col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt == 0, gt, predict)
    else:
        vis_predict = predict

    spy.save_rgb(save_predict_path, vis_predict, colors=colors)
    spy.save_rgb(save_gt_path, gt, colors=colors)