
# import student's functions
from solution import *
from sokoban import sokoban_goal_state, PROBLEMS
import os

#Select what to test
test_time_astar = True
test_time_gbfs = True
test_manhattan = True
test_fval_function = True
test_anytime_gbfs = False
test_alternate = False
test_anytime_weighted_astar = False

if test_time_astar:

  timebound = 5
  s0 = PROBLEMS[19] #Problems get harder as i gets bigger
  time = os.times()[0]
  weight = 2
  final = anytime_weighted_astar(s0, heur_fn=heur_alternate, weight=weight, timebound=timebound)
  difference = os.times()[0] - time
  print('This amount of time was given: {}'.format(timebound))
  print('Your anytime_weighted_astar used this amoung of time: {}'.format(difference))

  if difference - timebound < 0.1:
      print('Time keeping was acceptable on this problem')
  if difference - timebound >= 0.1:
      print('Time keeping was not acceptable on this problem')

if test_time_gbfs:

  timebound = 5
  s0 = PROBLEMS[19] #Problems get harder as i gets bigger
  time = os.times()[0]
  final = anytime_gbfs(s0, heur_fn=heur_alternate, timebound=timebound)
  difference = os.times()[0] - time
  print('This amount of time was given: {}'.format(timebound))
  print('Your anytime_gbfs used this amoung of time: {}'.format(difference))

  if difference - timebound < 0.1:
      print('Time keeping was acceptable on this problem')
  if difference - timebound >= 0.1:
      print('Time keeping was not acceptable on this problem')

if test_manhattan:
    ##############################################################
    # TEST MANHATTAN DISTANCE
    print('Testing Manhattan Distance')

    #Correct Manhattan distances for the initial states of the provided problem set
    correct_man_dist = [2, 8, 3, 3, 8, 7, 11, 11, 10, 12, 12, 13, 10, 13, 10, 35, 28, 41, 43, 36]

    solved = 0; unsolved = [];

    for i in range(0,20):
        #print("PROBLEM {}".format(i))

        s0 = PROBLEMS[i]

        man_dist = heur_manhattan_distance(s0)
        print('calculated man_dist:', str(man_dist))
        #To see state
        #print(s0.state_string())

        if man_dist == correct_man_dist[i]:
            solved += 1
        else:
            unsolved.append(i)

    print("*************************************")
    print("In the problem set provided, you calculated the correct Manhattan distance for {} states out of 20.".format(solved))
    print("States that were incorrect: {}".format(unsolved))
    print("*************************************\n")
    ##############################################################


if test_alternate:

  ##############################################################
  # TEST ALTERNATE HEURISTIC
  print('Testing alternate heuristic with best_first search')

  solved = 0; unsolved = []; benchmark1 = 7; benchmark2 = 15; timebound = 5 #time limit
  man_dist_solns = [6, 23, 24, 12, 24, -99, -99, 41, 20, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99]
  better_solns = [6, 23, 20, 12, 24, 32, -99, 41, 20, -99, 73, 52, 64, 39, 40, 160, 139, -99, -99, -99]
  for i in range(0, len(PROBLEMS)):

    print("*************************************")
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    se = SearchEngine('best_first', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_alternate)
    final, stats = se.search(timebound)

    if final:
      #final.print_path()
      solved += 1
    else:
      unsolved.append(i)

  print("\n*************************************")
  print("Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(len(PROBLEMS), solved, timebound))
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
  print("A heuristic that returns 0 solves 5 out of {} practice problems given {} seconds.".format(len(PROBLEMS),timebound))
  print("The manhattan distance implementation solved {} out of {} practice problems given {} seconds.".format(benchmark1,len(PROBLEMS),timebound))
  print("The better implementation solved {} out of {} practice problems given {} seconds.".format(benchmark2,len(PROBLEMS),timebound))
  print("*************************************\n")
  ##############################################################


if test_fval_function:

  test_state = SokobanState("START", 6, None, None, None, None, None, None, None)

  correct_fvals1 = [6, 11, 16]
  correct_fvals2 = [1594, 28, 16]
  correct_fvals3 = [24, 18, 16]

  ##############################################################
  # TEST fval_function
  print("*************************************")
  print('Testing fval_function')

  solved1 = 0
  weights = [0.01, .5, 1.]
  for i in range(len(weights)):

    test_node = sNode(test_state, hval=10, fval_function=fval_function)

    fval = round(fval_function(test_node, weights[i]),0)
    print ('Test', str(i), 'calculated fval:', str(fval), 'correct:', str(correct_fvals1[i]))

    if fval == correct_fvals1[i]:
      solved1 +=1

  print("\n*************************************")
  print("Your fval_function calculated the correct fval for {} out of {} tests.".format(solved1, len(correct_fvals1)))
  print("*************************************\n")

  ##############################################################


if test_anytime_gbfs:

  man_dist_solns = [4, 21, 18, 8, 18, -99, -99, 41, 15, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99]
  len_benchmark = [4, 21, 18, 8, 18, 31, -99, 41, 15, -99, 73, 45, 57, 39, 37, 160, 137, -99, 259, -99]

  ##############################################################
  # TEST ANYTIME GBFS
  print('Testing Anytime GBFS')

  solved = 0; unsolved = []; benchmark = 0; timebound = 5 #5 second time limit
  for i in range(0, len(PROBLEMS)):
    print("*************************************")
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    final = anytime_gbfs(s0, heur_fn=heur_alternate, timebound=timebound)

    if final:
      #final.print_path() #if you want to see the path
      if final.gval <= len_benchmark[i] or len_benchmark[i] == -99: #replace len_benchmark with man_dist_solns to compare with manhattan dist.
        benchmark += 1
      solved += 1
    else:
      unsolved.append(i)

  print("\n*************************************")
  print("Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(len(PROBLEMS), solved, timebound))
  print("Of the {} problems that were solved, the cost of {} matched or outperformed the benchmark.".format(solved, benchmark))
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
  print("The manhattan distance implementation solved 7 out of the 20 practice problems given 5 seconds.")
  print("The better implementation solved 16 out of the 20 practice problems given 5 seconds.")
  print("*************************************\n")

if test_anytime_weighted_astar:

  man_dist_solns = [4, 21, 10, 8, 17, -99, -99, 41, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99]
  len_benchmark = [4, 21, 10, 8, 18, 31, -99, 41, 14, -99, 36, 30, 28, 31, 27, -99, -99, -99, -99, -99]

  ##############################################################
  # TEST ANYTIME WEIGHTED A STAR
  print('Testing Anytime Weighted A Star')

  solved = 0; unsolved = []; benchmark = 0; timebound = 5 #5 second time limit
  for i in range(0, len(PROBLEMS)):
    print("*************************************")
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    weight = 10 #note that if you want to over-ride this initial weight in your implementation, you are welcome to!
    final = anytime_weighted_astar(s0, heur_fn=heur_alternate, weight=weight, timebound=timebound)

    if final:
      # final.print_path()
      print(final.gval, len_benchmark[i])
      if final.gval <= len_benchmark[i] or len_benchmark[i] == -99:
        benchmark += 1
      solved += 1
    else:
      unsolved.append(i)

  print("\n*************************************")
  print("Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(len(PROBLEMS), solved, timebound))
  print("Of the {} problems that were solved, the cost of {} matched or outperformed the benchmark.".format(solved, benchmark))
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
  print("The manhattan distance implementation solved 7 out of the 20 practice problems given 5 seconds.")
  print("The better implementation solved 13 out of the 20 practice problems given 5 seconds.")
  print("*************************************\n")
  ##############################################################

# def display(state: SokobanState):
#     boxes = [b for b in state.boxes]
#     obstacles = [o for o in state.obstacles]
#     storage = [s for s in state.storage]
#     robots = [r for r in state.robots]
#
#
#     for x in range(state.width):
#       for y in range(state.height):
#         if (x,y) in boxes:
#           print(' B ', end='')
#         elif (x,y) in obstacles:
#           print(' O ', end='')
#         elif (x,y) in storage:
#           print(' S ', end='')
#         elif (x,y) in robots:
#           print(' R ', end='')
#         else:
#           print(' _ ', end='')
#       print()

# for i in range(len(PROBLEMS)):
#
#   display(PROBLEMS[i])
#   print("\n\n")
