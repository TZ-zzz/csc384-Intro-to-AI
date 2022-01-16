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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        if successorGameState.isLose():
            return float('-inf')
        if action == Directions.STOP:
            return float('-inf')

        evaluation = 0.0
        foodList = newFood.asList()
        closestFood = min(manhattanDistance(newPos, food) for food in foodList) if foodList else 0
        closestGhost = min(manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
        shortest_scaredTime = min(newScaredTimes)
        if shortest_scaredTime > 0:
            evaluation += 0.5 / closestGhost
        else:
            evaluation -= 5 / closestGhost
            if closestGhost <= 1:
                return float('-inf')
        if newPos in currentGameState.getFood().asList():
            evaluation += float('inf')
        evaluation -= closestFood

        return evaluation


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def minimax(state, depth, index):
            best_move = Directions.STOP
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state), best_move

            if index == 0:
                value, nxt_depth = float('-inf'), depth
            elif index == state.getNumAgents() - 1:
                value, nxt_depth = float('inf'), depth - 1
            else:
                value, nxt_depth = float('inf'), depth

            for move in state.getLegalActions(index):
                nxt_pos = state.generateSuccessor(index, move)
                nxt_value, nxt_move = minimax(nxt_pos, nxt_depth, (index + 1) % state.getNumAgents())
                if index == 0 and value < nxt_value:
                    value, best_move = nxt_value, move
                if index != 0 and value > nxt_value:
                    value, best_move = nxt_value, best_move
            return value, best_move

        value, best_move = minimax(gameState, self.depth, self.index)
        return best_move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def AlphaBeta(state, depth, index, alpha, beta):
            best_move = Directions.STOP
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state), best_move

            if index == 0:
                value, nxt_depth = float('-inf'), depth
            elif index == state.getNumAgents() - 1:
                value, nxt_depth = float('inf'), depth - 1
            else:
                value, nxt_depth = float('inf'), depth

            for move in state.getLegalActions(index):
                nxt_pos = state.generateSuccessor(index, move)
                nxt_value, nxt_move = AlphaBeta(nxt_pos, nxt_depth, (index + 1) % state.getNumAgents(), alpha, beta)
                if index == 0:
                    if value < nxt_value:
                        value, best_move = nxt_value, move
                    if value >= beta:
                        return value, best_move
                    alpha = max(alpha, value)
                if index != 0:
                    if value > nxt_value:
                        value, best_move = nxt_value, best_move
                    if value <= alpha:
                        return value, best_move
                    beta = min(beta, value)
            return value, best_move

        value, best_move = AlphaBeta(gameState, self.depth, self.index, float("-inf"), float('inf'))
        return best_move


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

        def Expectimax(state, depth, index):
            best_move = Directions.STOP
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state), best_move

            if index == 0:
                value, nxt_depth = float('-inf'), depth
            elif index == state.getNumAgents() - 1:
                value, nxt_depth = 0, depth - 1
            else:
                value, nxt_depth = 0, depth

            moves = state.getLegalActions(index)
            for move in moves:
                nxt_pos = state.generateSuccessor(index, move)
                nxt_value, nxt_move = Expectimax(nxt_pos, nxt_depth, (index + 1) % state.getNumAgents())
                if index == 0 and value < nxt_value:
                    value, best_move = nxt_value, move
                if index != 0:
                    value += 1 / float(len(moves)) * float(nxt_value)
            return value, best_move

        value, best_move = Expectimax(gameState, self.depth, self.index)
        return best_move


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood().asList()
    curGhost = currentGameState.getGhostStates()
    curCapsules = currentGameState.getCapsules()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhost]
    evaluation = 0.0
    closestFood = min(manhattanDistance(curPos, food) for food in curFood) if curFood else 0
    closestGhost = min(manhattanDistance(curPos, ghost.configuration.pos) for ghost in curGhost)
    closestCapsules = min(manhattanDistance(curPos, capsule) for capsule in curCapsules) if curCapsules else 0
    shortest_scaredTime = min(curScaredTimes)

    # the important thing is to survive
    if shortest_scaredTime > 1:
        evaluation += 10 / closestGhost
    else:
        if closestGhost <= 1:
            return float('-inf')
        evaluation -= 20 / closestGhost

    # force it get closer to food
    evaluation -= closestFood + float(len(curFood))
    # force it get closer to capsules
    # evaluation -= closestCapsules + float(len(curFood))
    evaluation += currentGameState.getScore()
    evaluation += shortest_scaredTime * 0.5
    return evaluation


# Abbreviation
better = betterEvaluationFunction
