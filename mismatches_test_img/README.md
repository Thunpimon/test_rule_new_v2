# Mismatched Images Summary (from test_img)

This folder contains images that had prediction discrepancies during evaluation on `test_img` (50 images).

## Folder Structure

### 1. `01_rule_vs_ground_truth/` (13 images)
Images where the Physics Rule-Based Pipeline prediction did **NOT** match the Ground Truth label.
Grouped by category `GT_[TrueClass]__Pred_[PredictedClass]/`:
- **`GT_05_No_Star__Pred_01_Good/`**: 3 image(s)
- **`GT_02_Out_of_Focus__Pred_03_Tracking_Error/`**: 3 image(s)
- **`GT_01_Good__Pred_04_Over_Saturated/`**: 1 image(s)
- **`GT_01_Good__Pred_03_Tracking_Error/`**: 1 image(s)
- **`GT_01_Good__Pred_02_Out_of_Focus/`**: 1 image(s)
- **`GT_02_Out_of_Focus__Pred_01_Good/`**: 1 image(s)
- **`GT_06_Satellite__Pred_01_Good/`**: 1 image(s)
- **`GT_03_Tracking_Error__Pred_01_Good/`**: 1 image(s)
- **`GT_03_Tracking_Error__Pred_06_Satellite/`**: 1 image(s)

### 2. `02_rule_vs_cnn_disagreement/` (22 images)
Images where the Physics Rule-Based Pipeline and the CNN ONNX model disagreed with each other.
Grouped by category `Rule_[RulePrediction]__CNN_[CNNPrediction]/`:
- **`Rule_02_Out_of_Focus__CNN_01_Good/`**: 5 image(s)
- **`Rule_01_Good__CNN_05_No_Star/`**: 4 image(s)
- **`Rule_04_Over_Saturated__CNN_01_Good/`**: 4 image(s)
- **`Rule_01_Good__CNN_04_Over_Saturated/`**: 2 image(s)
- **`Rule_03_Tracking_Error__CNN_02_Out_of_Focus/`**: 2 image(s)
- **`Rule_01_Good__CNN_02_Out_of_Focus/`**: 1 image(s)
- **`Rule_03_Tracking_Error__CNN_01_Good/`**: 1 image(s)
- **`Rule_01_Good__CNN_06_Satellite/`**: 1 image(s)
- **`Rule_01_Good__CNN_03_Tracking_Error/`**: 1 image(s)
- **`Rule_06_Satellite__CNN_03_Tracking_Error/`**: 1 image(s)
