# nanomanufacturing

# Problem formulation
The goal of the project is to utilize machine leanring funcionts to analyze the motion of beads under the influence of voltage applied.

# Working environment
The project is based on OpenCV, and we use anaconda3 as primary working environment.
```sh
$ conda create --name [your working environment name] python=3.6
$ conda activate [your working environment name]
$ pip install opencv-python
```
# Run the program
1. Clone the repo from github.
```sh
$ git clone TODO
```
1. Include the two videos into this folder and rename as 10k20v.avi and 5mhz5v.avi.
2. Run python nano.py
```sh
$ python3 nano.py
```
3. The detected images are inside /10k20v and /5mhz5v folders
4. Result of features are 10k20v_output1/2.txt and 5mhz5v_output1/2.txt
5. To change parameters, you can add arguments directly through command line. For example, to adjust sampling interval, you can:
```sh
$ python3 nano.py --sampling_interval 3
```
