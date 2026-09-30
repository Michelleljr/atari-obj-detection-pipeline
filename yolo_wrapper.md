# JAXAtari Object-Centric Detection Pipeline

JAXAtari provides access to the ground-truth object-centric state directly from the simulator. In this work, we develop a method to obtain object-centric observations from object detection instead of using the ground-truth state.

`YOLOObjectCentricWrapper` provides vision-based object-centric observations while keeping the same interface as the other JAXAtari wrappers.

## Pipeline

```text
auto_explorer.py
      ↓
quirks_registry.json
      ↓
data_extractor.py
      ↓
YOLO training
      ↓
YOLOObjectCentricWrapper
```

| Step                       | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `auto_explorer.py`         | Discovers the object structure of each game.                                        |
| `quirks_registry.json`     | Stores the object and extraction information for each game.                         |
| `data_extractor.py`        | Extracts frames and creates labelled YOLO training data.                            |
| YOLO training              | Trains the object detection model using the extracted data.                         |
| `YOLOObjectCentricWrapper` | Uses the trained YOLO model to produce object-centric observations during gameplay. |

## Files

| File                       | Location                        | Description                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `quirks_registry.json`     | `praktikum_obj_detection_team/` | The main source of truth. It stores the mapping from object names to class IDs, coordinate offsets and padding, and the `detected_type` for each object. All other components use this information.                                                                |
| `auto_explorer.py`         | `praktikum_obj_detection_team/` | Runs a short exploration trajectory for each game. It checks the raw observation structure and creates an initial entry in `quirks_registry.json`. Object names are matched to the shared 7-class taxonomy. Objects that cannot be matched are marked as `REVIEW`. |
| `matrix_scanner.py`        | `praktikum_obj_detection_team/` | Diagnostic tool for inspecting the raw observation. It reads one frame and prints the object names and shapes. It can be used when bounding boxes do not look correct.                                                                                             |
| `game_patches.py`          | `praktikum_obj_detection_team/` | Contains custom OpenCV fallbacks for games that cannot be fully described by the registry. This includes cases such as missing RAM coordinates or objects that need color-based detection.                                                                         |
| `extractor_utils.py`       | `praktikum_obj_detection_team/` | Contains the generic RAM-array parsing functions: `extract_entity`, `extract_xy_pairs`, `extract_true_2d_grid`, and `extract_flat_per_row`.                                                                                                                        |
| `data_extractor.py`        | `praktikum_obj_detection_team/` | Main data extraction pipeline. It runs each game, reads the ground-truth object positions using the registry, and creates labelled YOLO data consisting of `.png` frames, `.txt` labels, and `classes.yaml`.                                                       |
| `YOLOObjectCentricWrapper` | `src/jaxatari/wrappers.py`      | JAXAtari-style wrapper that loads a trained YOLO checkpoint and the registry. It produces object-centric observations from live YOLO detections instead of the simulator ground truth.                                                                             |
| `Yolo_wrapper.py`          | `detectors/`                    | Interactive driver for testing the wrapper with a trained checkpoint. It supports keyboard controls and displays bounding boxes during gameplay.                                                                                                                   |
| `infer_yolov8n.py`         | `detectors/YOLOv8nano/`         | Standalone YOLO inference script. It supports static, live, and manual modes and can be used to test a checkpoint outside the JAXAtari wrapper.                                                                                                                    |

## Data Flow

The registry connects the ground-truth extraction and the YOLO-based wrapper.

```text
JAXAtari simulator
       │
       ▼
auto_explorer.py
       │
       ▼
quirks_registry.json
       │
       ├──────────────► data_extractor.py
       │                       │
       │                       ▼
       │                  YOLO dataset
       │                       │
       │                       ▼
       │                   YOLO training
       │                       │
       │                       ▼
       │                 trained .pt model
       │                       │
       └───────────────────────┤
                               ▼
                    YOLOObjectCentricWrapper
                               │
                               ▼
                  Object-centric observations
```

The simulator ground truth is only used to create the training labels. During normal use of `YOLOObjectCentricWrapper`, the object-centric observations are obtained from YOLO detections.
