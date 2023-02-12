# Interactive-Scribble-Segmentation
We present the code for our paper Interactive Scribble Segmentation (https://doi.org/10.7557/18.6823) presented at the Northen Lights Deep Learning conference 2023.

By drawing scribbles on an image our network will segment the class, which is being drawn, in real time. As such it is easy to correct the segmentation by simply adding more or less scribble.

![Motorcycle gif](https://github.com/MMLowes/Interactive-Scribble-Segmentation/blob/main/gifs/motercycle_gif.gif)


## Usage 
To run the annotation tool run the command
```
python demo_Qt.py --path /path/to/images
```
If no path is specified the default is `/images`. In the tool use the mouse to draw, and the network wil segment at each mouse movement. The keyboard shortcuts are:


