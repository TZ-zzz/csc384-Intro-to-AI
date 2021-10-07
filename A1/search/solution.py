#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
from search import *  # for search engines
from sokoban import SokobanState, Direction, \
    PROBLEMS  # for Sokoban specific classes and problems


def sokoban_goal_state(state):
    '''
  @return: Whether all boxes are stored.
  '''
    for box in state.boxes:
        if box not in state.storage:
            return False
    return True


def heur_manhattan_distance(state):
    # IMPLEMENT
    """admissible sokoban puzzle heuristic: manhattan distance"""
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of 
    the state to the goal. '''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    sum_distance = 0
    for box in state.boxes:
        nearest = float('inf')
        if box not in state.storage:
            for storage in state.storage:
                distance = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
                if distance < nearest:
                    nearest = distance
            sum_distance += nearest

    return sum_distance


# SOKOBAN HEURISTICS
def trivial_heuristic(state):
    '''trivial admissible sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state (# of moves required to get) to the goal.'''
    count = 0
    for box in state.boxes:
        if box not in state.storage:
            count += 1
    return count


def is_wall(position, state):
    return (position[0] < 0 or position[0] >= state.width) or (position[1] < 0 or position[1] >= state.width)


def is_wall_or_obs(position, state):
    return position in state.obstacles or is_wall(position, state)


def is_box(position, state):
    return position in state.boxes


def is_dead(state, box, s_x, s_y):
    top = (box[0], box[1] - 1)
    bot = (box[0], box[1] + 1)
    left = (box[0] - 1, box[1])
    right = (box[0] + 1, box[1])

    bot_ = is_wall_or_obs(bot, state)
    top_ = is_wall_or_obs(top, state)
    left_ = is_wall_or_obs(left, state)
    right_ = is_wall_or_obs(right, state)

    if (top_ or bot_) and (left_ or right_):
        return True

    if is_box(bot, state):
        bot_left = (bot[0] - 1, bot[1])
        bot_right = (bot[0] + 1, bot[1])
        if (left_ or right_) and (is_wall_or_obs(bot_left, state) or is_wall_or_obs(bot_right, state)):
            return True

    if is_box(top, state):
        top_left = (top[0] - 1, top[1])
        top_right = (top[0] + 1, top[1])
        if (left_ or right_) and (is_wall_or_obs(top_left, state) or is_wall_or_obs(top_right, state)):
            return True

    if is_box(left, state):
        bot_left = (left[0], left[1] + 1)
        top_left = (left[0], left[1] - 1)
        if (bot_ or top_) and (is_wall_or_obs(bot_left, state) or is_wall_or_obs(top_left, state)):
            return True
    if is_box(right, state):
        bot_right = (right[0], right[1] + 1)
        top_right = (right[0], right[1] - 1)

        if (bot_ or top_) and (is_wall_or_obs(bot_right, state) or is_wall_or_obs(top_right, state)):
            return True

    # check if one side against wall and no storage available for it
    if is_wall(top, state) or is_wall(bot, state):
        if box[1] not in s_y:
            return True
        num_boxes = 0
        num_storage = 0
        for box_ in state.boxes:
            if box_[0] == box[0]:
                num_boxes += 1
        for storage in state.storage:
            if storage[0] == box[0]:
                num_storage += 1
        if num_boxes < num_storage:
            return True

    if is_wall(left, state) or is_wall(right, state):
        if box[0] not in s_x:
            return True
        num_boxes = 0
        num_storage = 0
        for box_ in state.boxes:
            if box_[1] == box[1]:
                num_boxes += 1
        for storage in state.storage:
            if storage[1] == box[1]:
                num_storage += 1
        if num_boxes < num_storage:
            return True

    return False


# def need_change_d(state, robot):
#     top = (robot[0], robot[1] - 1)
#     bot = (robot[0], robot[1] + 1)
#     left = (robot[0] - 1, robot[1])
#     right = (robot[0] + 1, robot[1])
#
#     if is_box(top, state) and top not in state.storage:
#         if is_wall_or_obs((top[0], top[1] - 1), state) or is_box((top[0], top[1] - 1), state):
#             return True
#         elif (top[0] - 1, top[1]) in state.storage or (top[0] + 1, top[1]) in state.storage:
#             return True
#     if is_box(bot, state) and bot not in state.storage:
#         if is_wall_or_obs((bot[0], bot[1] + 1), state) or is_box((bot[0], bot[1] - 1), state):
#             return True
#         elif (bot[0] - 1, bot[1]) in state.storage or (bot[0] + 1, bot[1]) in state.storage:
#             return True
#     if is_box(left, state) and left not in state.storage:
#         if is_wall_or_obs((left[0] - 1, left[1]), state) or is_box((left[0] - 1, left[1]), state):
#             return True
#         elif (left[0], left[1] - 1) in state.storage or (left[0], left[1] + 1) in state.storage:
#             return True
#     if is_box(right, state) and right not in state.storage:
#         if is_wall_or_obs((right[0] + 1, right[1]), state) or is_box((right[0] + 1, right[1]), state):
#             return True
#         elif (right[0], right[1] - 1) in state.storage or (right[0], right[1] + 1) in state.storage:
#             return True


state_seen = {}


def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    global state_seen
    result = 0
    if state.parent is not None and state.boxes == state.parent.boxes:
        if state_seen[state.boxes] == float('inf'):
            return float('inf')

        result = state_seen[state.boxes]
        for box in state.boxes:
            if box not in state.storage:
                available_storage = set(state.storage).difference(set(state.storage) & set(state.boxes))
                storage_dis = [abs(box[0] - storage[0]) + abs(box[1] - storage[1]) for storage in available_storage]
                rob_dis = [abs(box[0] - robot[0]) + abs(box[1] - robot[1]) for robot in state.robots]
                result += min(rob_dis) + (min(storage_dis) / 2)

        return result

    state_seen[state.boxes] = 0
    s_x = [s[0] for s in state.storage]
    s_y = [s[1] for s in state.storage]
    for box in state.boxes:
        if box not in state.storage:
            if is_dead(state, box, s_x, s_y):
                state_seen[state.boxes] = float('inf')
                return float('inf')

            available_storage = set(state.storage).difference(set(state.storage) & set(state.boxes))
            storage_distance = [abs(box[0] - storage[0]) + abs(box[1] - storage[1]) for storage in available_storage]
            minimum = min(storage_distance)
            result += minimum
            state_seen[state.boxes] += minimum
            surrounding = ((box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), (box[0], box[1] - 1),
                           (box[0], box[1] + 1), (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1))
            result += len(set(state.obstacles) & set(surrounding))
            state_seen[state.boxes] += len(set(state.obstacles) & set(surrounding))

            robot_distance = [abs(box[0] - robot[0]) + abs(box[1] - robot[1]) for robot in state.robots]
            result += min(robot_distance)

    return result


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0


def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.
    Use this function stub to encode the standard form of weighted A* (i.e. g + w*h)
    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + weight * sN.hval


def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of anytime weighted astar algorithm'''
    weight = (initial_state.width + initial_state.height)
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    weighted_astar = SearchEngine('custom', 'full')
    weighted_astar.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
    # set the end time based on current time and timebound
    start_time = os.times()[0]
    end_time = start_time + timebound
    best = weighted_astar.search(timebound)
    if os.times()[0] >= end_time or best[0] is False:
        return best[0]

    costbound = (best[0].gval, best[0].gval, best[0].gval)
    while end_time > os.times()[0]:
        timebound = end_time - os.times()[0]
        if weight > 1:
            weight /= 2
        wrapped_fval_function = (lambda sN: fval_function(sN, weight))
        weighted_astar.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
        result = weighted_astar.search(timebound, costbound)
        if result[0] is not False:
            best = result
            costbound = (best[0].gval, best[0].gval, best[0].gval)

    return best[0]


def anytime_gbfs(initial_state, heur_fn, timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of anytime greedy best-first search'''
    # set the end time based on current time and timebound
    start_time = os.times()[0]
    end_time = start_time + timebound

    best_first = SearchEngine('best_first', 'full')
    best_first.init_search(initial_state, sokoban_goal_state, heur_fn)
    best = best_first.search(end_time - os.times()[0], None)
    # if the first search run out of time
    if os.times()[0] >= end_time or best[0] is False:
        return best[0]
    costbound = (best[0].gval, float('inf'), float('inf'))
    while end_time > os.times()[0]:
        timebound = end_time - os.times()[0]
        result = best_first.search(timebound, costbound)
        if result[0] is not False:
            best = result
            costbound = (best[0].gval, float('inf'), float('inf'))

    return best[0]
