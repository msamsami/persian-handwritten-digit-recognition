# persian-handwritten-digit-recognition-system-pso-pnn
Persian Handwritten Digit Recognition System using Particle Swarm Optimization Based K-Means Clustering and Probabilistic Neural Network

This project is a MATLAB simulation of the method presented by [Mehran Taghipour-GorjiKolaie et al.](https://www.semanticscholar.org/paper/Persian-Handwritten-Digit-Recognition-Using-Swarm-MehranTaghipour-GorjiKolaie-Miri/375c8fce8a3eada82aa3fea145e28c1da64087ff) in 2014. In order to recognize Persian (Farsi) handwritten digits, a combination of intelligent clustering method and probabilistic neural network (PNN) has been utilized. [Hoda data set](http://farsiocr.ir/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%AF%D8%A7%D8%AF%D9%87/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%A7%D8%B1%D9%82%D8%A7%D9%85-%D8%AF%D8%B3%D8%AA%D9%86%D9%88%DB%8C%D8%B3-%D9%87%D8%AF%DB%8C/), which contains 80000 Persian handwritten digit images, has been used to evaluate the classifier.

![Screenshot](block-diagram.jpg)

 The advantages of the method can be mentioned as follow:
- The PNN is one of the most powerful classifiers which maps information and adapt itself by training data. Many applications using this classifier reports its ability to present high recognition rate for training data.
- The big problem of PNN is when it encounters a large number of training data, therefore, extraction of feature vectors using clustering is used.
- Center of each cluster is an average of features in cluster and can be a appropriate representation of the cluster. Center of each cluster is used as training feature vectors.
- Each class of Persian handwritten digits has its own nature and it is impossible to use same number of clusters for them. Thus, partical swarm optimization (PSO) technique is employed to find the optimum number of clusters for each class (or in other word, optimum feature vectors for training step).

More details are explained in the paper.
