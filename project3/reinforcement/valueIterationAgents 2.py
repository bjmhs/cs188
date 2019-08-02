# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for num in range(self.iterations):
            track = util.Counter()
            for s in self.mdp.getStates():
                maxVal = -1000000
                for a in self.mdp.getPossibleActions(s):
                    maxVal = max(self.computeQValueFromValues(s, a), maxVal)
                    track[s] = maxVal
            self.values = track


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        T = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        gamma = self.discount

        for new_state, probability in T:
            R = self.mdp.getReward(state, action, new_state)
            qValue += probability * (R + gamma * self.values[new_state])
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestA = None
        maxVal = -1000000

        for a in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, a)
            if qVal > maxVal:
                maxVal = qVal
                bestA = a
        return bestA

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for num in range(self.iterations):
            s = states[num % len(states)]
            if not self.mdp.isTerminal(s):
                qVals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                self.values[s] = max(qVals)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pred = {}                             #predecessors
        priorityqueue = util.PriorityQueue()  #initialize priority queue

        for s in self.mdp.getStates():        #compute predecessors of all states
            if not self.mdp.isTerminal(s):
                for a in self.mdp.getPossibleActions(s):
                    T = self.mdp.getTransitionStatesAndProbs(s, a)
                    for new_state, probability in T:
                        if new_state not in pred:
                            pred[new_state] = {s}
                        else:
                            pred[new_state].add(s)

        """Beyond this point: follow pseudocode given in Q5 for project"""
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                qVals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                abs_diff = abs(max(qVals) - self.values[s])
                priorityqueue.update(s, -abs_diff)

        for iteration in range(self.iterations):
            if not priorityqueue.isEmpty():
                s = priorityqueue.pop()
                if not self.mdp.isTerminal(s):
                    qVals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                    self.values[s] = max(qVals)

                for predecessor in pred[s]:
                    if not self.mdp.isTerminal(predecessor):
                        qVals = [self.computeQValueFromValues(predecessor, a) for a in self.mdp.getPossibleActions(predecessor)]
                        abs_diff = abs(max(qVals) - self.values[predecessor])
                        if abs_diff > self.theta:
                            priorityqueue.update(predecessor, -abs_diff)




