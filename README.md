# Chess Recording Entity Position Estimator (CReEPEr)

Readme and project reproducability under construction.

## About
This project was started for a Hunter College Summer 2025 Capstone Course, developed by Ishraq Mahid, Warren Wu, Ivan Huang, and Tafseer Haque over the course of 2 months.

The goal of the project was to create a model capable of taking a 2D image of a chessboard with chess pieces and outputting a representation of the chess positions. We utilize YOLO for object detection to detect the pieces, and create/utilize custom algorithms for board segmentation and creating the corresponding mappings.
![Architecture](/architecture.jpg)

## Data and Testing
Due to the lack of labeled data on chess boards and piece segmentations, we wrote data-generation scripts to automatically create Blender renders of a chessboard set in various positions. This allows the development of a proof of concept while having some data to work with. While synthetic data is obviously flawed, by using a synthetic dataset we could reasonably make progress on the scale of a class-project and with the time constraints. However, it still served as a realistic basis for the project and enabled development in a way which techniques could realistically be scaled onto real world data.

With this synthetic data, we wrote out test scripts to test each individual stage of the model pipeline.



## YOLO
For chess piece and board detection, we trained a YOLO model on the annotated synthetic dataset and obtained the following results.

![ConfusionMatrix](/yolo_training/confusion_matrix.png)

![PRCurve](/yolo_training/BoxPR_curve.png)

![Example](/yolo_training/yolo_WP_detect.png)

*Example detection on white pawn.*

## Board Segmentation
We approached board segmentation from the perspective of corner annotations. By using the corners, we can identify and label the orientation of the 8x8 grid. However, given the nature of different perspectives, slight changes in the corners can result in vastly different square locations, prompting us to utilize perspective transformation as an intermediate step in the IoU evaluation.

![GT_and_Predicted](/evaluation/Original%20Image_screenshot_29.07.2025.png)

*Green is ground truth; blue is predicted.*

![Corner2Mask](/evaluation/Predicted%20Mask_screenshot_29.07.2025.png)

*Predicted square segmentation masks.*

We utilized an algorithm developed by samobot over the course of the project, and developed our own corner detection algorithm with promising results.

The following is the IoU results of samobot's algorithm:
![ChessboardIoUs](/evaluation/board_segmentation/avg_score_by_angle.png)

## Piece to Board Mappings
With object detections and board segmentation from corners, we utilize a brute-force implementation to map the piece to the board, marking a point closer to the bottom of the piece, and using the point's location on a corresponding square segmentation to evaluate the location of the piece. The following is the result on ground truth segmentations and masks.

![Piece2Board](/evaluation/board_mappings/pass_rate_by_angle.png)