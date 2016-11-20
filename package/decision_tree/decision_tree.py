import operator
import math

#Reference : http://blog.csdn.net/dream_angel_z/article/details/45965463
class MyClass:
    def __init__(self):
        return

    def createDataSet(self):
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing','flippers']   #the label of each feature
        #change to discrete values
        return dataSet, labels

    def calcShannonEnt(self,dataSet):
        n = len(dataSet) #calculate the size of dataset
        labelCounts = {}
        # create dictionary "count"
        for featVec in dataSet: #the the number of unique elements and their occurance
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/n #notice transfering to float first
            shannonEnt -= prob * math.log(prob,2) #log base 2
        return shannonEnt

    def splitDataSet(self,dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self,dataSet):
        numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
        baseEntropy = self.calcShannonEnt(dataSet)  #calculate the info of dataSet
        bestInfoGain = 0.0; bestFeature = -1
        for i in range(numFeatures):        #iterate over all the features
            featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
            uniqueVals = set(featList)       #get a set of unique values
            newEntropy = 0.0
            for value in uniqueVals:         # calculate the info of each feature
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
            if (infoGain > bestInfoGain):       #compare this to the best gain so far
                bestInfoGain = infoGain         #if better than current best, set to best
                bestFeature = i
        return bestFeature                      #returns an integer

    def majorityCnt(self,classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self,dataSet,labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]#stop splitting when all of the classes are equal
        if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value),subLabels)
        return myTree

    def classify(self,inputTree,featLabels,testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else: classLabel = valueOfFeat
        return classLabel


def main():
    x = MyClass()
    dataSet,labels=x.createDataSet()
    #Since label will be changed during tree building, os labels2 is a copy
    dataSet,labels2=x.createDataSet()

    print dataSet,labels
    myTree= x.createTree(dataSet,labels)
    print myTree

    #labels is the name of feature, [1,0] is test sample
    print x.classify(myTree,labels2,[1,1])

if __name__ == "__main__":
    main()