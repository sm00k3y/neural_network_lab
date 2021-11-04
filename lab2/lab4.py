import numpy as np
from numpy.core.fromnumeric import shape
import idx2numpy
from matplotlib import pyplot as plt
sigma = .03

"""
param a - wyjścia z neruonów poprzedniej warstwy, Nx1
param w - wagi, MxN
param bias - wektor biasów, Mx1
return: całkowite pobudzenie, Mx1
"""
def calculateTotalStimulation(a, w, bias):
    return np.dot(w, a) + bias
    

"""
param f - funkcja aktywacji
z - całkowite pobudzenie, Mx1
return: wyjścia z neuronów, Mx1
"""
def calculateNeuronsOutput(f, z):
    return f(z)



"""
param shapes - liczba neuronów w kolejnych warstwach
"""
def generateWsBiases(shapes):
    ws = []
    biass = []
    for i in range(len(shapes)-1):
        ws.append(np.random.normal(0, sigma, (shapes[i+1], shapes[i])))
        biass.append(np.random.normal(0, sigma, (shapes[i+1], 1)))

    return (ws, biass)


"""
param x - wejscie,Nx1
param ws - lista wag, Lx[,]
param biass - lista biasów, Lx[,]
param f - funkcja aktywacji
return: wyjscie, Mx1
"""
# def calculateOutput(x, ws, biass, f):
#     for i in range(len(ws) - 1):
#         x = calculateTotalStimulation(x, ws[i], biass[i])
#         x = calculateNeuronsOutput(f, x)
#     x = calculateTotalStimulation(x, ws[len(ws) - 1], biass[len(ws) - 1])
#     return x

"""
param x - wejscie,Nx1
param ws - lista wag, Lx[,]
param biass - lista biasów, Lx[,]
param fs - funkcje aktywacji, L
return: wyjscie, Mx1
"""
def calculateOutputVariousFActivation(x, ws, biass, fs):
    ass = []
    zs = []
    for i in range(len(ws)):
        x = calculateTotalStimulation(x, ws[i], biass[i])
        zs.append(x)
        x = calculateNeuronsOutput(fs[i], x)
        ass.append(x)
    return (x, ass, zs)


def softmax(x):
    x = np.exp(x)
    x /= np.sum(x)
    return x

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def htangens(x):
    return np.tanh(x)

def reLu(x):
    return np.maximum(x, 0)

def nothing(x):
    return x


"""
param a - wartość f(x), wyjście z neuronu
"""
def softmaxDer(a):
    return a * (1-a)

def sigmoidDer(a):
    return a * (1-a)

def htangensDer(a):
    return 1 - a*a

def reLuDer(a):
    return (a > np.zeros(a.shape)) * 1.0

def nothingDer(a):
    return 1.0

########## LAB4

def calculateOutputError(yPredicted, yReal):
    return -(yReal - yPredicted)

"""
param a - wyjście z neuronu, Nx1
param w - wagi z warstwą następną, MxN 
param delta2 - błędy warstwy następnej, Mx1
param fDer - pochodna f. aktywacji (param: f(x))
return: błędy, Nx1
"""
def calculateHiddenError(a, w, delta2, fDer):
    return fDer(a) * np.dot(np.transpose(w), delta2)


"""
param yPredicted - wyjście przewidywane
param yReal - wyjście rzeczywiste
param ass - wyjścia neuronów poszczególnych warstw
param ws - wagi w poszczególnych warstwach
param fDers - pochodne f. aktywacji w poszczególnych warstwach
    (param: f(x))

"""
def calculateDeltaWsBiases(yPredicted, yReal, ass, ws, fDers):
    ass.reverse()
    ws.reverse()
    fDers.reverse()
    errors = []
    deltaWs = []
    error = calculateOutputError(yPredicted, yReal)
    errors.append(error)
    deltaWs.append(np.dot(error, np.transpose(ass[1])))
    for i in range(len(ws) - 1):
        error = calculateHiddenError(ass[i+1], ws[i], error, fDers[i+1])
        errors.append(error)
        deltaWs.append(np.dot(error, np.transpose(ass[i+2])))
    deltaWs.reverse()
    errors.reverse()
    ass.reverse()
    ws.reverse()
    fDers.reverse()
    return(deltaWs, errors)


def updateWsBiases(ws, biass, deltaWss, errorss, alpha):
    for j in range(len(errorss[0])):
        errorsAvg = errorss[0][j] - errorss[0][j]
        deltaWsAvg = deltaWss[0][j] - deltaWss[0][j]
        for i in range(len(errorss)):
            errorsAvg += errorss[i][j] / len(errorss)
            deltaWsAvg += deltaWss[i][j] / len(deltaWss)
        ws[j] = ws[j] - alpha * deltaWsAvg
        biass[j] = biass[j] - alpha * errorsAvg
    return (ws, biass)

def performLearningMiniBatch(xs, ys, ws, biass, fs, fDers, alpha):
    deltaWss = []
    errorss = []
    for i in range(len(xs)):
        output, ass, _ = calculateOutputVariousFActivation(xs[i], ws, biass, fs)
        ass.insert(0, xs[i])
        deltaWs, errors = calculateDeltaWsBiases(output, ys[i], ass, ws, fDers)
        deltaWss.append(deltaWs)
        errorss.append(errors)
    return updateWsBiases(ws, biass, deltaWss, errorss, alpha)

def performLearningEpoch(xs, ys, ws, biass, fs, fDers, alpha, batchSize):
    xss = chunks(xs, batchSize)
    yss = chunks(ys, batchSize)
    
    for i in range(len(xss)):
        ws, biass = performLearningMiniBatch(xss[i], yss[i], ws, biass, fs, fDers, alpha)
    return ws, biass


def chunks(l, n):
    n = max(1, n)
    return list((l[i:i+n] for i in range(0, len(l), n)))


def printCorrectFraction(x, y, ws, biass, fs):
    correctPred = 0
    for i in range(len(x)):
        output = calculateOutputVariousFActivation(x[i], ws, biass, fs)
        if np.argmax(y[i]) == np.argmax(output[0]):
            correctPred += 1
    print("correct", correctPred/len(x))


dataset = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
expectations = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

y = np.array([[[1] if expectations[i] == j else [0] for j in range(10)] for i in range(len(expectations))])

dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1] * dataset.shape[2], 1))/np.max(dataset)
x = dataset[0]

fs = [sigmoid, sigmoid, softmax]
fDers = [sigmoidDer, sigmoidDer, softmaxDer]
shapes = [x.shape[0], 300, 100, 10]

ws, biass = generateWsBiases(shapes)
# biass = [np.array([[.3] for i in range(5)]), np.array([[.5] for i in range(3)]), np.array([[.2] for i in range(10)])]
# for i in range(len(ws)):
#     for j in range(len(ws[i])):
#         for k in range(len(ws[i][j])):
#             ws[i][j][k] = i * (j+k) / 1000

alpha = .2
batchSize = 20


# output = calculateOutputVariousFActivation(x, ws, biass, fs)
# print(output[0])
# print(np.sum(output[0]))

max_len = 1000
x = dataset[0:max_len]
y = y[0:max_len]
for i in range(20):
    printCorrectFraction(x, y, ws, biass, fs)
    ws, biass = performLearningEpoch(x, y, ws, biass, fs, fDers, alpha, batchSize)

printCorrectFraction(x, y, ws, biass, fs)

# correctPred = 0

# for i in range(max_len):
#     output = calculateOutputVariousFActivation(x[i], ws, biass, fs)
#     print("+++++")
#     print(output[0])
#     print(np.sum(output[0]))
#     print(expectations[i], np.argmax(output[0]))
#     if expectations[i] == np.argmax(output[0]):
#         correctPred += 1
    
#     # first_image = x[i]
#     # first_image = np.array(first_image, dtype='float')
#     # pixels = first_image.reshape((28, 28))
#     # plt.imshow(pixels, cmap='gray')
#     # plt.show()

# print("correct", correctPred/max_len)

correctPred = 0
val_len = 100
x = dataset[max_len : max_len + val_len]
for i in range(val_len):
    output = calculateOutputVariousFActivation(x[i], ws, biass, fs)
    print("+++++")
    print(output[0])
    print(np.sum(output[0]))
    print(expectations[i+max_len], np.argmax(output[0]))
    if expectations[i+max_len] == np.argmax(output[0]):
        correctPred += 1
    
    # first_image = x[i]
    # first_image = np.array(first_image, dtype='float')
    # pixels = first_image.reshape((28, 28))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

print("correct", correctPred/val_len)

