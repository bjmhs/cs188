# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newScore = successorGameState.getScore()
        oldFoodCount = currentGameState.getFood().count()
        value = 0

        if oldFoodCount == newFood.count():
            value = closestFoodDist(newPos, newFood)
        else:
            value = 0
        for ghost in newGhostStates:
            value += 4 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
        return -value

        # avgGhostDistance = avgGhostDist(newPos, newGhostStates)
        # closestFoodDistance = closestFoodDist(newPos, newFood)
        # avgFoodDistance = avgFoodDist(newPos, newFood)

        "*** YOUR CODE HERE ***"


def avgFoodDist(PacmanPos, newFood):
    "Returns the average food distance"
    distSum = count = 0

    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y]:
                distSum += manhattanDistance(PacmanPos, [x, y])
                count += 1

    return distSum/count

def avgGhostDist(PacmanPos, newGhostStates):
    "Returns the average distance from Pacman's position to ghosts"
    distSum = count = 0
    for ghost in newGhostStates:
        distSum += manhattanDistance(PacmanPos, ghost.getPosition())
        count += 1
    return distSum/count

def closestFoodDist(PacmanPos, newFood):
    "Returns the distance to the closest food from Pacman"
    dist = []
    for food in newFood.asList():
        dist.append(manhattanDistance(PacmanPos, food))
    return min(dist)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def ghostMin(index, state, depth):

            if depth < 0 or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), None]

            value = 1000000
            min = None

            if index >= state.getNumAgents() - 1:
                for action in state.getLegalActions(index):
                    successor = state.generateSuccessor(index, action)
                    score = pacmanMax(0, successor, depth)[0]
                    if score <= value:
                        min = action
                        value = score

            else:
                for action in state.getLegalActions(index):
                    successor = state.generateSuccessor(index, action)
                    score = ghostMin(index + 1, successor, depth)[0]
                    if score <= value:
                        min = action
                        value = score

            return [value, min]

        def pacmanMax(index, state, depth):

            depth -= 1
            if depth < 0 or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), None]

            value = -1000000
            max = None

            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                score = ghostMin(index + 1, successor, depth)[0]
                if score >= value:
                    max = action
                    value = score
            return [value, max]

        return pacmanMax(0, gameState, self.depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def ghostMin(index, state, depth, alpha, beta):

            if depth < 0 or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), None]

            value = 1000000
            minA = None

            if index >= state.getNumAgents() - 1:
                for action in state.getLegalActions(index):
                    successor = state.generateSuccessor(index, action)
                    score = pacmanMax(0, successor, depth, alpha, beta)[0]
                    if score <= value:
                        minA = action
                        value = score
                    if value < alpha:
                        return [value, minA]
                    beta = min(beta, value)
            else:
                for action in state.getLegalActions(index):
                    successor = state.generateSuccessor(index, action)
                    score = ghostMin(index + 1, successor, depth, alpha, beta)[0]
                    if score <= value:
                        minA = action
                        value = score
                    if value < alpha:
                        return [value, minA]
                    beta = min(beta, value)

            return [value, minA]

        def pacmanMax(index, state, depth, alpha, beta):

            depth -= 1
            if depth < 0 or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), None]

            value = -1000000
            maxA = None

            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                score = ghostMin(index + 1, successor, depth, alpha, beta)[0]
                if score >= value:
                    maxA = action
                    value = score
                if value > beta:
                    return [value, maxA]
                alpha = max(alpha, value)
            return [value, maxA]

        return pacmanMax(0, gameState, self.depth, -1000000000000, 1000000000000)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
