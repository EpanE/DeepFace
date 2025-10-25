# DeepFace Utilities

This repository collects a number of experimental scripts and utilities that build on top of the [DeepFace](https://github.com/serengil/deepface) framework for face recognition, facial attribute analysis, and attendance tracking.  The code base mixes desktop GUIs, Raspberry Pi prototypes, and OpenCV command line tools that showcase different inference pipelines for the DeepFace models.

## Repository layout

The project is organised around several standalone entry points.  The most notable ones are:

| Script | Description |
| ------ | ----------- |
| `DeepFace-UI.py` | PyQt5 desktop application with tabs for face analysis, verification, and database search using interchangeable DeepFace backends. |
| `DF_Live_Opencv.py`, `DF_Live_RetinaFace.py` | Live camera demos that combine OpenCV capture with RetinaFace detection and DeepFace embeddings. |
| `DP_App_*.py` | Raspberry Pi oriented prototypes for attendance kiosks, including simplified IoT-ready versions. |
| `Compile_FaceEmbedings.py` | Helper that builds a pickled embedding index from images under `dataset/` for rapid lookup. |
| `DeepFace1.py`, `DeepFace2.py` | Minimal examples showing how to invoke DeepFace verification and analysis routines. |

Supporting assets such as enrolment images live under `dataset/` while generated embeddings are stored as `face_embeddings*.pkl`.  Attendance logs are persisted as CSV files in the project root.

## Requirements

Most scripts target Python 3.8+ and depend on the following core libraries:

- `deepface`
- `opencv-python`
- `tensorflow`
- `numpy`
- `pandas`
- `PyQt5`
- `Pillow`

Some variants also rely on device specific packages (e.g. GPIO libraries on Raspberry Pi).  Consult the individual scripts before running them on constrained hardware.

## Getting started

1. Create and activate a virtual environment.
2. Install the dependencies.  A quick starting point is:

   ```bash
   pip install -r requirements.txt  # create this file based on your environment
   ```

   Alternatively, replicate the packages from `piplist.txt` if you are matching the original development environment.
3. Place enrolment images inside the `dataset/` directory (each person in a separate folder) and, if needed, pre-compute embeddings via `Compile_FaceEmbedings.py`.
4. Launch the script that matches your use case.  For example, to open the GUI application:

   ```bash
   python DeepFace-UI.py
   ```

   or to run the OpenCV live demo:

   ```bash
   python DF_Live_Opencv.py
   ```

## Tips

- Many scripts expect a working camera connected to the machine.
- The GUI tools provide options to switch between multiple DeepFace backends to balance accuracy and speed.
- For production deployments, review the code for proper error handling and persistence before use.

## Contributing

Pull requests are welcome.  If you add a new experiment or hardware integration, accompany it with documentation so others can reproduce your setup.

