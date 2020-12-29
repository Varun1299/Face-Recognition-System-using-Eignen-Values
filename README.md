# Face-Recognition-System-using-Eignen-Values



### Dependencies

- Python 3.4
- OpenCV
- Numpy

### How to Run?

There are two ways to run the program:

- Recognize faces for all test images

```shell
python main.py
```

- Recognizes face for a specified image

```
python main.py <path_to_image>
```



### Output Format

- When we recognize faces for all test images, the output is a table in which the first column specifies the expected output, the second column specifies the actual output and the third column wheter the recognition was correct or not. 

- When we recognize face in a specified image . The output ranges from s1 till s40.

  For example, If the output is s4 then that means that the test image is recognized to be the person whose images are in folder s4.


The methodology is as described in the following papers:

[Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86](https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71)


[Eigenfaces for Face Detection/Recognition](https://www.cse.unr.edu/~bebis/MathMethods/PCA/case_study_pca1.pdf)

**For details on dataset used and results, please refer to Report.pdf**

---

_This work was done as part of assignment in the Probability and Random Processes course at IIT Gandhinagar._


