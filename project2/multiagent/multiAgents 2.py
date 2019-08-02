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

        if newFood.count() == currentGameState.getFood().count():
            score = 10000000
            foodDistances = [manhattanDistance(food, newPos) for food in newFood.asList()]
            score = min(min(foodDistances), score)
        else:
            score = 0

        ghostDistances = [2 ** (7 - manhattanDistance(ghost.getPosition(), newPos)) for ghost in newGhostStates]
        score += sum(ghostDistances)
        return score * -1



        """ ORIGINAL HARDCODE
        score = 0
        INCREMENT = 10
        DECREMENT = -10
        scareTime = min(newScaredTimes)
        if action == "STOP":
            score += (20 * DECREMENT)
        else:
            # eat food?
            if (newFood.count() == 1 or scareTime >= 3):
                score += (20 * INCREMENT)
            else:
                fDistanceNew = foodDistance(newPos, newFood)
                fDistanceCurr = foodDistance(currentPosition, newFood)
                ghostDistanceNew = ghostDistance(newPos, newGhostStates)
                ghostDistanceCurr = ghostDistance(currentPosition, newGhostStates)
                if newFood.count() == foodCount:
                    score += 0
                elif fDistanceNew < fDistanceCurr:
                    score += INCREMENT
                else:
                    score += DECREMENT
                #ghost distance
                if ghostDistanceNew > ghostDistanceCurr:
                    score += INCREMENT
                else:
                    score += DECREMENT
                #ghost vs food distance
                if fDistanceNew - ghostDistanceNew > fDistanceCurr - ghostDistanceCurr:
                    score += 0.5 * INCREMENT
                else:
                    score += 0.5 * DECREMENT
        return score"""

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
        totalAgents = gameState.getNumAgents()
        depth = self.depth * totalAgents - 1
        counter = 0
        a = ""

        def val(s, a, counter):
            if depth <= counter or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            elif a != 0:
                return expectVal(s, a, counter)
            elif a == 0:
                return maxVal(s, a, counter)

        def maxVal(s, a, counter):
            max_val = -100000000
            counter += 1
            for action in s.getLegalActions(a):
                state = s.generateSuccessor(a, action)
                max_val = max(max_val, val(state, (a + 1) % totalAgents, counter))
            return max_val

        def expectVal(s, a, counter):
            expect_val = 0
            counter += 1
            for action in s.getLegalActions(a):
                state = s.generateSuccessor(a, action)
                expect_val += val(state, (a + 1) % totalAgents, counter)
            expect_val = expect_val / len(s.getLegalActions(a))
            return expect_val

        legalActions = gameState.getLegalActions(0)
        scores = []
        x = 0

        for a in legalActions:
            state = gameState.generateSuccessor(x, a)
            scores.append((val(state, (x+1) % totalAgents, counter), a))

        a = max(scores, key = lambda item: item[0])
        return a[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currCapsules = currentGameState.getCapsules()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    scaredState = False

    if max(currScaredTimes) > 0:
        scaredState = True

    ghostDistances = [manhattanDistance(currPos, ghost.getPosition()) for ghost in currGhostStates]
    foodDistances = [manhattanDistance(currPos, food) for food in currFood.asList()]
    capsuleDistances = [manhattanDistance(currPos, capsule) for capsule in currCapsules]

    fScore = 0
    if (len(currFood.asList()) > 0):
        fScore = 1 / float(min([currFood.width + currFood.height] + foodDistances))
    else:
        fScore = 1

    cScore = 0
    if (len(currCapsules) > 0):
        cScore = 1 / float(min([len(currCapsules)] + capsuleDistances))
    else:
        cScore = 1

    gScore = 0
    if (min(ghostDistances) < 1):
        gScore = -100
    else:
        gScore = 1 / float(min(ghostDistances))

    fWeight = 1
    gWeight = 4
    cWeight = 3

    if ((min(ghostDistances) < max(currScaredTimes)) and scaredState):
        ghostWeight = 100
        gScore = abs(gScore)

    return (fWeight * fScore) + (gWeight * gScore) + (cWeight * cScore) + currentGameState.getScore()



# Abbreviation
better = betterEvaluationFunction
