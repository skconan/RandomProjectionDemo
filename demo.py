from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def demo():
    digits = datasets.load_digits()
    # N = 1797
    # original_d = 
    # print(dir(digits))
    n, original_dimension = digits.data.shape
    accuracies = []
    components = np.int32(np.linspace(2, 64, 20))
    print()
    print("="*40)
    print("The number of observation:",n)
    print("Dimensional of original data:",original_dimension)
    print("Dimensional of new data:",components)
    print("="*40)
    
    # SVM
    split = train_test_split(digits.data, digits.target, test_size = 0.3, random_state = 42)
    (trainData, testData, trainTarget, testTarget) = split
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    print("Baseline accuracy:",baseline)
    # johnson_lindenstrauss_min_dim(N,eps=0.1)

    

    print("Random projection accuracies")

    # loop over the projection sizes
    for comp in components:
        # create the random projection
        sp = SparseRandomProjection(n_components = comp)
        X = sp.fit_transform(trainData)
    
        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)
    
        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        acc = metrics.accuracy_score(model.predict(test), testTarget)
        accuracies.append(acc)
        print(comp,":",acc)

    # create the figure
    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on Digits")
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    plt.xlim([2, 64])
    plt.ylim([0, 1.0])
    
    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(accuracies), color = "r")
    plt.plot(components, accuracies)

    plt.show()

if __name__=='__main__':
    demo()