import math
from random import randint, random, choices, seed
import matplotlib.pyplot as plt

#Set up the Q values of the cliffworld
Qvals = []
#Q value array is size of world * directions
for row in range(4):
    Qvals.append([])
    for column in range(12):
        Qvals[row].append([])
        for direction in range(4):
            Qvals[row][column].append(0)

#Define a function to reset all Q values
def resetQvals():
    for row in range(4):
        for column in range(12):
            for direction in range(4):
                Qvals[row][column][direction] = 0
            
#Define movement behaviour, movement is defined as 0 up, 1 right, 2 down, 3 left 
def move(tile, movement):
    posX, posY = tile[0], tile[1]
    nextPos = None
    if movement == 0:
        if posY == 0: 
            nextPos = [posX, posY]
        else:
            nextPos = [posX, posY - 1]
    if movement == 1:
        if posX == 11: 
            nextPos = [posX, posY]
        else:
            nextPos = [posX + 1, posY]
    if movement == 2:
        if posY == 3: 
            nextPos = [posX, posY]
        else:
            nextPos = [posX, posY + 1]
    if movement == 3:
        if posX == 0: 
            nextPos = [posX, posY]
        else:
            nextPos = [posX - 1, posY]
    #Move back if you fall into the cliff
    if nextPos[1] == 3 and (nextPos[0] > 0 and nextPos[0] < 11):
        nextPos = [0, 3]
        nextPos.append(-100)
    #Add reward for moving not into the cliff
    else:
        nextPos.append(-1)
    #Returning the next position and the corresponding reward given     
    return nextPos

#Define action strategy functions
def greedyHelp(actions):
    maxIndex = 0
    maxAction = actions[maxIndex]
    for index in range(len(actions)):
        if actions[index] > maxAction:
            maxAction = actions[index]
            maxIndex = index
    return maxIndex

def egreedyHelp(actions, epsilon):
    if random() < epsilon:
        return randint(0, 3)
    maxIndex = 0
    maxAction = actions[maxIndex]
    for index in range(len(actions)):
        if actions[index] > maxAction:
            maxAction = actions[index]
            maxIndex = index
    return maxIndex
    
def softmaxHelp(actions, beta):
    denominator = 0
    chancesList = []
    for index in range(len(actions)):
        denominator += math.e ** (beta * actions[index])
        
    for index in range(len(actions)):
        chancesList.append(math.e ** (beta * actions[index]) / denominator)
    possibleAction = [0,1,2,3]
    return choices(possibleAction, weights=chancesList)[0]

#Setting pu the algorithms
#Define variables
startPosX, startPosY = 0, 3
endPosX, endPosY = 11, 3
trials = 500
alpha, gamma, epsilon, beta = 0.5, 1, 0.10, 2
actionStrategy = "e-greedy" #The options are: random, greedy, e-greedy or softmax

def Qlearning():
    resetQvals()
    rewardCounters = []
    for n in range(trials):
        posX, posY = startPosX, startPosY
        rewardCounter = 0
        while posX != endPosX or posY != endPosY:
            #Choosing the next action
            action = None
            if actionStrategy == "random":
                action = randint(0, 3)
            elif actionStrategy == "greedy":
                action = greedyHelp(Qvals[posY][posX])
            elif actionStrategy == "e-greedy":
                action = egreedyHelp(Qvals[posY][posX], epsilon)
            elif actionStrategy == "softmax":
                action = softmaxHelp(Qvals[posY][posX], beta)
                
            #Execute action    
            newPosX, newPosY, reward = move([posX, posY], action)
            
            #Update qVals
            Qvals[posY][posX][action] = Qvals[posY][posX][action] + alpha * (reward + (gamma * max(Qvals[newPosY][newPosX])) - Qvals[posY][posX][action])
            
            #Keep track of total reward of episode
            rewardCounter += reward
            #Update position
            posX, posY = newPosX, newPosY
        rewardCounters.append(rewardCounter)
    return rewardCounters
        
def sarsa():    
    resetQvals()
    rewardCounters = []
    for n in range(trials):
        posX, posY = startPosX, startPosY
        rewardCounter = 0
        #Choose action for start position
        action = None
        if actionStrategy == "random":
            action = randint(0, 3)
        elif actionStrategy == "greedy":
            action = greedyHelp(Qvals[posY][posX])
        elif actionStrategy == "e-greedy":
            action = egreedyHelp(Qvals[posY][posX], epsilon)
        elif actionStrategy == "softmax":
            action = softmaxHelp(Qvals[posY][posX], beta)
        
        while posX != endPosX or posY != endPosY:
            #Execute action
            newPosX, newPosY, reward = move([posX, posY], action)
            #Keep track of total reward of episode
            rewardCounter += reward
            
            #Choose action for next state
            newAction = None
            if actionStrategy == "random":
                newAction = randint(0, 3)
            elif actionStrategy == "greedy":
                newAction = greedyHelp(Qvals[newPosY][newPosX])
            elif actionStrategy == "e-greedy":
                newAction = egreedyHelp(Qvals[newPosY][newPosX], epsilon)
            elif actionStrategy == "softmax":
                newAction = softmaxHelp(Qvals[newPosY][newPosX], beta)
            #update qVals based on next action
            Qvals[posY][posX][action] = Qvals[posY][posX][action] + alpha * (reward + (gamma * Qvals[newPosY][newPosX][newAction]) - Qvals[posY][posX][action])
            
            #update position and action
            posX, posY = newPosX, newPosY
            action = newAction
        rewardCounters.append(rewardCounter)    
    return rewardCounters
#Fix seed

seed(1)
#Create a normalize function to improve the graphs
#Parameter for normalizing
maxDeviation = 300
def normalize1(results):
    meanOfResults = sum(results)/len(results)
    minOfResults = min(results)
    maxOfResults = max(results)
    deviationPerDifference = 0
    #The min result is more extreme than the max result
    if meanOfResults - minOfResults > maxOfResults - meanOfResults:
        deviationPerDifference = maxDeviation / (meanOfResults - minOfResults)
    else:
        deviationPerDifference = maxDeviation / (maxOfResults - meanOfResults)
    normalizedResults = []
    for result in results:
        normalizedResults.append(meanOfResults + (result - meanOfResults) * deviationPerDifference)
    return(normalizedResults)
    
#Create another normalize function for different graphs
groupsize = 10
def normalize2(results):
    normalizedResults = []
    for index in range(len(results)):
        currentGroup = 0
        sumOfCurrentGroup = 0
        while index + currentGroup < len(results) and currentGroup < 10:
            sumOfCurrentGroup += results[index+currentGroup]
            currentGroup += 1
        normalizedResults.append(sumOfCurrentGroup / currentGroup)
    return normalizedResults
    
#Create another normalize function for combining both normalize function
def normalizeCombine(results):
    return normalize2(normalize1(results))

#Make the plots
QlearningResults = Qlearning()
sarsaResults = sarsa()
plt.plot(normalizeCombine(QlearningResults), linewidth = 0.7, label="Q-learning")
plt.plot(normalizeCombine(sarsaResults), linewidth = 0.7, label="sarsa")
plt.legend()
plt.ylim(-100, 0)
plt.xlim(0, 500)
plt.ylabel("Sum of rewards of episode")
plt.xlabel("Episode")
plt.title("Normalized results of sarsa and Q-learning using e-greedy with epsilon=0.1")
plt.show()
