# ST2FG
See requirements.txt to setup an environment required to execute the code.
To run the code, run the jupyter notebook: st2f_test.ipynb

## For Inference:
Follow the instructions provided in the .ipynb file. Cell no. 3 has the necessary parameters defined to run the inference code and cell no. 4 contains the commmand line code to run the inference. The flag '--mode' can be changed to 'gen' for generating images based on the provided description or left as 'man' for manipulating the input image based on the description.

The Python script st2f.py which is used in the .ipynb file has additional optional parameters that can be printed out by simply typing:
> python st2f.py --help

Although it is recommended to simply use the st2f_inference.ipynb script as it includes the code snippet to download the trained models at the correct path.

The results will be stored at the path './results/test' and the target/test images for manipulation must be a .jpg file and stored in a directory './examples/'
