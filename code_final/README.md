# Simple_animal_tracking

## Introduction

This folder contains the project report, source codes, a command line executable file allow users to use the proposed tracing method to trace  their own animal movement videos. **The language version used for codes is python 3.8.5.**

- **threshold_tracing.py:**  The threshold-based tracing module 
- **camshift_tracing.py:**  The camshift-based tracing module
- **evaluation.py:**  The performance evluation/comparision codes
- **trace.py:**  User command line executable file, use this as a tool to generate your own trajectory tracing  output in a probability csv! For implementation details, please see below:

**Overview of the package used**
matplotlib                         3.3.2
scikit-image                       0.17.2
numpy                              1.20.3
pandas                             1.1.3


## Getting Started

1.unzip the zip in a directory 
if you want make a new directory for it 

```text
mkdir code_test
```

```text
unzip code_package.zip
```

2.Run the commandline (e.g black_1_short.mp4)
```text 
#to use thresholding tracing method
python3 trace.py -i black_1_short.mp4  -t True -t1 220 -t2 255 -o outputfilename

#to use Camshift tracing methdods
python3 trace.py -i black_1_short.mp4 -cs True  -o outputfilename 
```

3.Get the tracing result!
