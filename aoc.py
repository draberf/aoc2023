import argparse
import sys
import os
import re

ADVANCED = False
OUTPUT = []

def dayOne(input):

    digits = {
        "one":1,
        "two":2,
        "three":3,
        "four":4,
        "five":5,
        "six":6,
        "seven":7,
        "eight":8,
        "nine":9
    }
    rev_digits = {k[::-1]: v for k, v in digits.items()}
    ord0 = ord('0')
    ord9 = ord('9')

    def processLine(line, replace=False):

        def processSide(line, dict=digits):

            if line == "":
                raise Exception("line completely exhausted")

            if replace:
                for k, v in dict.items():
                    if len(line) < len(k):
                        continue
                    if line[:len(k)] == k:
                        return v

            if (c := ord(line[0])) >= ord0 and c <= ord9:
                return int(line[0])

            return processSide(line[1:], dict)
        
        out = 10 * processSide(line) + processSide(line[::-1], dict=rev_digits)
        print(out)
        OUTPUT.append(f"{line} {out}")
        return out
    
    return sum([processLine(line, replace=ADVANCED) for line in input])

def dayTwo(input):
    
    maxcolor = {
        "red":12,
        "green":13,
        "blue":14
    }

    def processGroup(group, mins):
        for pull in group.split(", "):
            num, color = pull.split(" ")
            try:
                mins[color] = max(int(num), mins[color])
            except Exception as e:
                print(mins)
                raise(e)
        return mins

    def processLine(line):
        gametag, groups = line.split(": ")
        id = int(gametag[5:])
        valid = True
        mins = {
            "red":0,
            "green":0,
            "blue":0
        }
        for group in groups.split("; "):
            mins = processGroup(group, mins)
            if valid:
                for color, count in mins.items():
                    if count > maxcolor[color]:
                        valid = False
                        break

        if ADVANCED: return mins["red"]*mins["green"]*mins["blue"]
        return id if valid else 0
    
    return sum([processLine(line) for line in input])

def dayThree(input):
    
    # outline:
    # process line by line
    # look for numbers 1-9
    # find longest trail of numbers, get its start and end coord
    # search surroundings for a non-number, non-period
    # if you find, add number to sum

    WIDTH = len(input[0])
    HEIGHT = len(input)
    
    GEARS = {}

    def isZero(c):
        return (ord(c) == ord('0'))
    def isNonzeroNum(c):
        return (o := ord(c)) >= ord('1') and o <= ord('9')
    def isNum(c):
        return isZero(c) or isNonzeroNum(c)

    def pingGear(row, col, num):
        index = f"{row}:{col}"
        if index not in GEARS.keys():
            GEARS[index] = []
        GEARS[index].append(num)

    def processGears():
        ratio_sum = 0
        for pings in GEARS.values():
            if len(pings) == 2:
                ratio_sum += pings[0] * pings[1]
        return ratio_sum


    def searchRegionForSymbol(x1, y1, x2, y2, num=0):
        valid = False
        for row in range(max(y1, 0), min(y2+1, HEIGHT)):
            for col in range(max(x1, 0), min(x2+1, WIDTH)):
                c = input[row][col]
                if c == '.': continue
                if isNum(c): continue
                if c == '*': pingGear(row, col, num)
                valid = True
        return valid

    def scanLine(line, y):
        total = 0
        counting = False
        number = 0
        for x, c in enumerate(line):

            if not counting:
                if isZero(c):
                    raise Exception("leading zero encountered on line", line)
                if isNonzeroNum(c):
                    number = int(c)
                    counting = True
            else:
                if isNum(c):
                    number = number*10 + int(c)

                if not isNum(c) or x+1 == WIDTH:
                    l = len(str(number))
                    if searchRegionForSymbol(x-l-1, y-1, x, y+1, number): total += number
                    counting = False
            
        return total

    part_sum = sum([scanLine(line, y) for y, line in enumerate(input)])
    gear_sum = processGears()
    return gear_sum if ADVANCED else part_sum

def dayFour(input):

    card_wins = len(input)*[1]

    def processLine(i, line):
        _, win_nums_s, got_nums_s  = re.match(r"^(.*): (.*) \| (.*)$", line).groups()

        win_field = 100*[False]

        for win_num in [int(num) for num in win_nums_s.split()]:
            win_field[win_num] = True

        card_val = 0

        for got_num in [int(num) for num in got_nums_s.split()]:
            if win_field[got_num]: card_val += 1

        if card_val == 0: return 0

        # add cards to card wins
        for j in range(card_val):
            card_wins[i+j+1] += card_wins[i]

        return 2**(card_val-1)

    if ADVANCED:
        for i, line in enumerate(input):
            processLine(i, line)
        return sum(card_wins)
    return sum([processLine(i, line) for (i, line) in enumerate(input)])

def dayFive_temp(input):


    def processMap(in_nums, map):

        out_nums = []


        for num in in_nums:

            foundMap = False
            for s1, s2, s3 in [line.split() for line in map]:

                dest_start = int(s1)
                source_start = int(s2)
                length = int(s3)

                if num >= source_start and num < (source_start + length):
                    out_nums.append(dest_start + num - source_start)
                    foundMap = True
                    break
            
            if not foundMap: out_nums.append(num)

        return out_nums

    seeds = [int(s) for s in input[0][7:].split()]
    input.pop(0)

    mapScanning = False
    currMap = []

    while input:
        currLine = input[0]
        input.pop(0)

        if not mapScanning and currLine[-5:] == " map:":
            mapScanning = True
            continue

        if not mapScanning and currLine == "":
            continue

        if mapScanning and currLine == "":
            mapScanning = False
            seeds = processMap(seeds, currMap)
            currMap = []
            continue

        if mapScanning:
            currMap.append(currLine)
            continue

        raise Exception("Unrecognized state-input combination", mapScanning, currLine)

    return min(seeds)

def dayFive(input):

    def isOverlap(intval1: (int, int), intval2: (int, int)) -> bool:
        start1, end1 = intval1
        start2, _ = intval2

        if start1 <= start2:
            return end1 >= start2
        
        else:
            return isOverlap(intval2, intval1)

    def mergeIntervals(intval1: (int, int), intval2: (int, int)) -> (int, int):
        start1, end1 = intval1
        start2, end2 = intval2
        return (min(start1, start2), max(end1, end2))

    def addToIntervalArray(arr: [(int, int)], intval: (int, int)) -> [(int, int)]:

        if not arr: return [intval]

        first = arr[0]

        if isOverlap(intval, first):
            return addToIntervalArray(arr[1:], mergeIntervals(intval, first))
        else:
            return [first] + addToIntervalArray(arr[1:], intval)
        
    
    # return: list of tuples of intervals and whether they've been processed, or left intact
    def mapInterval(intval : (int, int), mapping: (int, int, int)) -> [((int, int), bool)]:
        
        start, end = intval
        dest_start, src_start, length = mapping
        src_end = src_start + length - 1

        if not isOverlap((start, end), (src_start, src_end)):
            return [(intval, False)]
        
        results = []

        if start < src_start:
            results.append(((start, src_start-1), False))

        # processed segment
        processed_start = max(start, src_start)
        processed_end = min(end, src_end)
        results.append((
            (dest_start + processed_start-src_start, dest_start + processed_end - src_start),
            True
        ))

        if end > src_end:
            results.append(((src_end+1, end), False))

        return results


    seeds = [int(s) for s in input[0][7:].split()]
    intervals = []
    processed_intervals = []
    for i in range(0, len(seeds), 2):
        processed_intervals = addToIntervalArray(processed_intervals, (seeds[i],seeds[i]+seeds[i+1]-1))

    seed_tags = []

    for line in input[2:]:

        if line[-4:] == "map:":
            seed_tags = len(seeds)*[False]
            for new_intval in processed_intervals:
                intervals = addToIntervalArray(intervals, new_intval)
            print(line, intervals)
            processed_intervals = []
            continue

        if line == "":
            continue

        # numbers
        dest_start, src_start, length = [int(val) for val in line.split()]

        for i, seed in enumerate(seeds):
            if seed_tags[i]: continue
            if seed >= src_start and seed <= src_start + length:
                seeds[i] = dest_start + (seed - src_start)
                seed_tags[i] = True

        # process intervals
        untouched_intervals = []
        for intval in intervals:
            results = mapInterval(intval, (dest_start, src_start, length))
            for interval, processed_flag in results:
                if processed_flag:
                    processed_intervals = addToIntervalArray(processed_intervals, interval)
                else:
                    untouched_intervals = addToIntervalArray(untouched_intervals, interval)
        
        intervals = untouched_intervals.copy()


    print(seeds)
    res_intervals = intervals + processed_intervals
    print(res_intervals)
    if ADVANCED: return min([start for start, _ in res_intervals])
    return min(seeds)

def daySix(input):

    from math import sqrt, floor, ceil

    def solveRace(total, best):
        # solve x for x^2 - total*x + best
        sqrtd = sqrt(total**2 - 4*best)
        return (
            0.5 * (total - sqrtd),
            0.5 * (total + sqrtd)
        )
    
    if ADVANCED:
        total = int("".join(input[0][10:].split()))
        best  = int("".join(input[1][10:].split()))
        x1, x2 = solveRace(total, best+0.001)
        dof = floor(x2) - ceil(x1) + 1
        print(total, best, x1, x2, dof)
        return dof


    race_total = [int(v) for v in input[0][10:].split()]
    race_best  = [int(v) for v in input[1][10:].split()]

    product = 1

    for total, best in zip(race_total, race_best):
        x1, x2 = solveRace(total, best+0.001)
        dof = floor(x2) - ceil(x1) + 1
        print(total, best, x1, x2, dof)
        product *= dof

    return product

def daySeven(input):

    from functools import cmp_to_key

    cardList = "AKQT98765432J" if ADVANCED else "AKQJT98765432"

    def cardStrength(card: str) -> int:
        return -cardList.index(card)

    def handStrength(hand: [str]) -> int:
        
        handDict = {c: hand.count(c) for c in cardList}

        jHandDict = handDict.copy()

        if ADVANCED:
            j_count = handDict["J"]
            for k in jHandDict.keys():
                if k == "J": continue
                jHandDict[k] += j_count

        def fiveOfAKind() -> bool:
            return 5 in handDict.values()
        
        def fourOfAKind() -> bool:
            return 4 in handDict.values()
        
        def fullHouse() -> bool:
            return 3 in handDict.values() and 2 in handDict.values()
        
        def threeOfAKind() -> bool: 
            return 3 in handDict.values()
        
        def twoPair() -> bool:
            pairs = 0
            for val in handDict.values():
                if (pairs := pairs + (1 if val == 2 else 0)) >= 2:
                    return True
            return False
        
        def onePair() -> bool:
            return 2 in handDict.values()
        
        def highCard() -> bool:
            return True

        handTypes = [
            fiveOfAKind, fourOfAKind, fullHouse,
            threeOfAKind, twoPair, onePair, highCard
        ]

        for i, type in enumerate(handTypes):
            if type(): return -i

    def replaceJoker(hand: [str]) -> int:

        if "*" not in hand: return handStrength(hand)

        new_hand = hand.replace("*","",1)
        return max([replaceJoker(new_hand+c) for c in cardList])

    def cmpEqualStrengthHand(hand1: [str], hand2: [str]) -> int:
        
        if (card1 := hand1[0]) == (card2 := hand2[0]): return cmpEqualStrengthHand(hand1[1:], hand2[1:])

        return cardStrength(card1) - cardStrength(card2)
    
    class Game:


        def __init__(self, hand: str, bid: str):
            self.hand = hand
            self.strength = handStrength(hand)
            if ADVANCED:
                self.strength = max(self.strength, replaceJoker(hand.replace("J", "*")))
            self.bid = int(bid)

        def __repr__(self) -> str:
            return f"hand: {self.hand} strength: {self.strength} bid: {self.bid}"

    def cmpGame(game1: Game, game2: Game) -> int:

        if game1.strength == game2.strength:
            return cmpEqualStrengthHand(game1.hand, game2.hand)
        
        return game1.strength - game2.strength


    games = [Game(hand, bid) for hand, bid in [line.split() for line in input]]
    games = sorted(games, key=cmp_to_key(cmpGame))
    total = sum([game.bid*(rank+1) for rank, game in enumerate(games)])
    
    return total

def dayEight(input):

    node_mapping = 17576*[""]
    
    from math import lcm

    def nodeToInt(node: [str]):
        f = lambda x: ord(x) - ord('A')
        index = f(node[0]) * 26**2 + f(node[1]) * 26 + f(node[2])
        node_mapping[index] = node
        return index

    def intToNode(index: int):
        return node_mapping[index]

    curr_node = nodeToInt('AAA')
    end_node = nodeToInt('ZZZ')
    adv_starter_nodes = []

    instructions = input[0]

    left  = (end_node+1)*[-1]
    right = (end_node+1)*[-1]


    for node_line in input[2:]:
        node, new_left, new_right = re.match(r"^([A-Z]{3}) = \(([A-Z]{3}), ([A-Z]{3})\)$", node_line).groups()
        index = nodeToInt(node)
        if node[2] == "A": adv_starter_nodes.append(index)
        left[index] = nodeToInt(new_left)
        right[index] = nodeToInt(new_right)

    class Ghost():

        def __init__(self, start_node):
            self.curr_node = start_node

        def move(self, instruction):
            if instruction == "L":
                self.curr_node = left[self.curr_node]
            else:
                self.curr_node = right[self.curr_node]

        def isDone(self) -> bool:
            if ADVANCED: return intToNode(self.curr_node)[2] == "Z"
            else: return intToNode(self.curr_node) == "ZZZ"

    ghosts = [Ghost(nodeToInt("AAA"))] if not ADVANCED else [Ghost(node) for node in adv_starter_nodes]
    ghost_finishes = []

    for ghost in ghosts:

        done = False
        steps = 0
        while not done:
            for instruction in instructions:

                if ghost.isDone():
                    ghost_finishes.append(steps)
                    done = True
                    break

                ghost.move(instruction)

                steps += 1

    return lcm(*ghost_finishes)
    
def dayNine(input):
    
    def diffSequence(sequence: [int]) -> [int]:
        if len(sequence) < 2: raise Exception("list too short to make differences")
        return [sequence[i+1]-sequence[i] for i in range(len(sequence)-1)]


    def predictSequence(sequence: [int]) -> int:
        if all([val == 0 for val in sequence]): return 0

        if not ADVANCED:
            return sequence[-1] + predictSequence(diffSequence(sequence))
        else:
            return sequence[0] - predictSequence(diffSequence(sequence))
    
    sequences = [[int(n) for n in line.split()] for line in input]
    return sum([predictSequence(sequence) for sequence in sequences])

def dayTen(input):
    
    pipes = {
        "|": {'N': 'N', 'S': 'S'},
        "-": {'E': 'E', 'W': 'W'},
        "J": {'E': 'N', 'S': 'W'},
        "L": {'S': 'E', 'W': 'N'},
        "F": {'N': 'E', 'W': 'S'},
        "7": {'N': 'W', 'E': 'S'}
    }

    WIDTH = len(input[0])
    HEIGHT = len(input)

    def getDirection(dirIn: str, pipe: str) -> str:
        pipe_dirs = pipes[pipe]
        if dirIn in pipe_dirs.keys(): return pipe_dirs[dirIn]
        raise Exception("Pipe entered from a wrong direction")
    
    def move(coord: (int,int), direction: str) -> (int,int):
        x, y = coord
        if direction == 'N':
            return (x, y-1)
        if direction == 'E':
            return (x+1, y)
        if direction == 'W':
            return (x-1, y)
        if direction == 'S':
            return (x, y+1)
        
    start_y = [i for i, line in enumerate(input) if "S" in line][0]
    start_x = input[start_y].index("S")

    loop_only = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
    loop_only[start_y][start_x] = "S"

    x : int
    y : int
    direction : str

    # find starting pipe tile
    # check North
    if input[start_y-1][start_x] in "|7F":
        x = start_x
        y = start_y
        direction = 'N'
    # check East
    elif input[start_y][start_x+1] in "-J7":
        x = start_x+1
        y = start_y
        direction = 'E'
    # check West
    elif input[start_y][start_x-1] in "-LF":
        x = start_x-1
        y = start_y
        direction = 'W'
    # check South
    else:
        x = start_x
        y = start_y + 1
        direction = 'S'

    # for inside/outside
    start_direction = direction


    # walk
    steps = 1
    while (tile := input[y][x]) != "S":

        loop_only[y][x] = tile

        direction = getDirection(direction, tile)
        x, y = move((x, y), direction)
        steps += 1

    if not ADVANCED: return steps//2

    start_pipe = "S"

    for pipe, directions in pipes.items():
        for in_dir, out_dir in directions.items():
            if in_dir == direction and out_dir == start_direction: start_pipe = pipe

    loop_only[start_y][start_x] = start_pipe 

    for line in loop_only:
        OUTPUT.append("".join(line))

    def scanLine(line: str) -> int:
        is_inside = False
        is_pipe = False
        entered_down = False
        inside = 0
        processed_line = "" 
        try:
            for c in line:

                if c == " ": processed_line += "I" if is_inside else "O"
                else: processed_line += c

                if not is_pipe:
                    if c == " ":
                        inside += 1 if is_inside else 0
                        continue
                    if c == "|": 
                        is_inside = not is_inside
                        continue
                    if c == "L":
                        is_pipe = True
                        entered_down = False
                        continue
                    if c == "F":
                        is_pipe = True
                        entered_down = True
                        continue
                    raise Exception("Wrong character encountered")
                
                else:
                    if c == "-":
                        continue
                    if c == "J":
                        is_pipe = False
                        if entered_down: is_inside = not is_inside
                        continue
                    if c == "7":
                        is_pipe = False
                        if not entered_down: is_inside = not is_inside
                        continue
                    raise Exception("Wrong character encountered")

        except Exception as e:
            print("ERROR", processed_line)
            raise e
        
        return inside

    return sum([scanLine(line) for line in loop_only])

def dayEleven(input):

    # expand galaxies
    orig_width = len(input[0])
    orig_height = len(input)

    expansion_factor = 2 if not ADVANCED else 1000000

    expand_rows = []
    expand_cols = []

    for i, line in enumerate(input):
        if "#" not in line: expand_rows.append(i)
    
    for i, col in enumerate(zip(*input)):
        if "#" not in col: expand_cols.append(i)

    # find all galaxies

    galaxies = []
    for y, line in enumerate(input):
        for x, c in enumerate(line):
            if c == "#": galaxies.append((x, y))

    paths = 0

    # count shortest paths
    for i, (x1, y1) in enumerate(galaxies):
        for ii, (x2, y2) in enumerate(galaxies[i+1:]):
            path = y2 - y1 + abs(x2 - x1)

            expanded_cols = [col for col in expand_cols if col > min(x1,x2) and col < max(x1,x2)]         
            expanded_rows = [row for row in expand_rows if row > min(y1,y2) and row < max(y1,y2)]         

            ext_path = (len(expanded_cols) + len(expanded_rows)) * (expansion_factor-1)

            paths += path + ext_path

    return paths

def dayTwelve(input):

    PRINT = False

    def processSprings(springs: str, groups: [int]) -> int:


        # stop conditions
        if not groups:
            if '#' in springs: return 0
            return 1
        
        group_len = sum(groups) + len(groups) - 1

        if len(springs) < group_len:
            return 0

        # find largest group
        maxgroup, max_i = max(groups), groups.index(max(groups))

        pre, post = groups[:max_i], groups[max_i+1:]
        pre_len = 0 if not pre else sum(pre) + len(pre) - 1
        post_len = 0 if not post else sum(post) + len(post) - 1

        if (maxgroup+1)*"#" in springs:
            return 0


        acc = 0

        for i in range(pre_len, len(springs)-maxgroup-post_len+1):
            if "." not in springs[i:i+maxgroup]:
                if i > 0:
                    if springs[i-1] == '#': continue
                if i+maxgroup+1 < len(springs):
                    if springs[i+maxgroup+1] == '#': continue
                
                # split L - grp - R
                # print(pre, springs[:i-1], maxgroup*"#", springs[i+maxgroup+1:], post)
                acc += \
                    processSprings(springs[:i-1], pre) * \
                    processSprings(springs[i+maxgroup+1:], post)
            
        return acc



    def processLine(line) -> int:
        springs, groups = line.split()
        groups = [int(n) for n in groups.split(',')]

        if ADVANCED:
            groups = 5*groups
            springs = "?".join(5*(springs))

        res = processSprings(springs, groups)

        return res

    results = 0

    for line in input:
        res = processLine(line)
        print(line, res)
        results += res

    return results

def dayThirteen(input):
    
    def findOneDifference(line1, line2) -> int:
        if line1 == line2: return -1

        diff = 0
        index = 0

        for i, (c1, c2) in enumerate(zip(line1, line2)):
            if c1 != c2:
                diff += 1
                index = i
            if diff > 1: return -1
        return index

    def findSymmetry(group) -> int:
        
        # i -> indicates *after* which line we show symmetry
        for i in range(len(group)-1):
            valid = True
            smudgeFound = False
            for ii in range(1+min(i, len(group)-i-2)):
                if group[i-ii] != group[i+ii+1]:
                    if ADVANCED and not smudgeFound:
                        if findOneDifference(group[i-ii], group[i+ii+1]) > -1:
                            smudgeFound = True
                            continue
                    valid=False
                    break

            if valid and (smudgeFound == ADVANCED):
                return i + 1
        return 0    

    def checkGroupForSymmetries(group) -> int:

        t_group = ["".join(list(tuple)) for tuple in zip(*group)]
        return 100 * findSymmetry(group) + findSymmetry(t_group)

    group = []
    acc = 0

    for line in input:
        if not line:
            acc += checkGroupForSymmetries(group)
            group = []
        else:
            group.append(line)
        
    acc += checkGroupForSymmetries(group)

    return acc

def dayFourteen(input):

    def tiltLine(line):

        spheres = 0
        for i, c in enumerate(line):
            if c == "#":
                return spheres*"O" + (i - spheres)*"." + "#" + tiltLine(line[i+1:])
            if c == "O": spheres += 1

        return spheres*"O" + (len(line)-spheres)*"."      

    def tiltEast(platform):
        
        return [tiltLine(line) for line in platform]

    def tiltAny(platform, direction): 
        
        if direction == 'E':
            return tiltEast(platform)
        
        if direction == 'W':
            rev = [line[::-1] for line in platform]
            return [line[::-1] for line in tiltAny(rev, 'E')]
        
        if direction == 'N':
            tpose = ["".join(line) for line in zip(*platform)]
            return ["".join(line) for line in zip(*tiltAny(tpose, 'E'))]
        
        if direction == 'S':
            return tiltAny(platform[::-1], 'N')[::-1]
        
        raise Exception
    
    def scoreNorth(platform):

        return sum([(len(platform)-i) * sum([1 for c in line if c == "O"]) for i, line in enumerate(platform)])

    def scoreAny(platform, direction):

        if direction == 'N':
            return scoreNorth(platform)

    if not ADVANCED:
        tilted = (tiltAny(input, 'N'))
        return scoreAny(tilted, 'N')
    
    def match(platform1, platform2):
        return all([line1 == line2 for line1, line2 in zip(platform1, platform2)])

    tilted = input
    intermediates = []
    finished = False
    cycle_start = -1
    cycle_len = -1
    CHECK_START = 700
    for i in range(1000000000):
        

        if i >= CHECK_START:
            print(i)
            for ii, interm in enumerate(intermediates):
                if match(tilted, interm):
                    print("Match found between", i, "and", CHECK_START+ii)
                    print("values:")
                    for iii, interm_2 in enumerate(intermediates[ii:]):
                        print(ii+iii, scoreAny(interm_2, "N"))
                    print(i-CHECK_START, scoreAny(tilted, 'N'))
                    cycle_start = CHECK_START+ii
                    cycle_len = i-cycle_start
                    print(cycle_start, cycle_len)
                    finished = True
                    break
            intermediates.append(tilted)

        if finished: break

        tilted = tiltAny(tilted, 'N')
        tilted = tiltAny(tilted, 'E')
        tilted = tiltAny(tilted, 'S')
        tilted = tiltAny(tilted, 'W')


    return scoreAny(intermediates[(1000000000 - cycle_start) % cycle_len], 'N')

def dayFifteen(input):

    boxes = [({}, []) for _ in range(256)]

    def processGroup(group: str) -> int:

        current_value = 0
        for c in group:
            current_value += ord(c)
            current_value *= 17
            current_value %= 256
        return current_value
    
    def processInstruction(group: str) -> None:
        label, op, focus = re.match(r"([a-z]+)(-|=)([1-9]?)", group).groups()
        hash = processGroup(label)
        if op == "-":
            boxes[hash][0][label] = 0
            if label in boxes[hash][1]:
                boxes[hash][1].remove(label)
        if op == "=":
            if label not in boxes[hash][1]:
                boxes[hash][1].append(label)
            boxes[hash][0][label] = int(focus)

    def focusingPower(box: (dict, [str]), box_id) -> int:
        
        focal_lengths, order = box

        acc = 0

        for i, label in enumerate(order):
            acc += (i+1)*focal_lengths[label]
            print(label, "box", box_id, "slot", i+1, "fl", focal_lengths[label], "res", (i+1)*focal_lengths[label])


        return (box_id+1) * acc

        
    if not ADVANCED:
        return sum([processGroup(group) for group in input[0].split(",")])

    for group in input[0].split(","):
        processInstruction(group)

    return sum([focusingPower(box, i) for i, box in enumerate(boxes)])

def daySixteen(input):

    WIDTH = len(input[0])
    HEIGHT = len(input)

    dir_dicts = {
        "/":  {"N": "E", "E": "N", "W": "S", "S": "W"},
        "\\": {"N": "W", "E": "S", "W": "N", "S": "E"}
    }

    # create direction field, only checked at splitters
    energized = [[False for _ in range(WIDTH)] for _ in range(HEIGHT)]

    def clearEnergized():
        for y in range(HEIGHT):
            for x in range(WIDTH):
                energized[y][x] = False


    def printEnergized():
        for row in energized:
            print("".join(["#" if cell else "." for cell in row]))

    def countEnergized():
        acc = 0
        for row in energized:
            acc += sum([1 if cell else 0 for cell in row])
        return acc

    class Beam:

        x: int
        y: int
        direction: str

        def __init__(self, x, y, direction):
            self.x = x
            self.y = y
            self.direction = direction

        def checkEnd(self) -> bool:
            if self.x < 0 or self.x >= WIDTH: return True
            if self.y < 0 or self.y >= HEIGHT: return True

            # in bounds, not on a splitter
            if input[self.y][self.x] not in "|-": return False

            # on a splitter
            if energized[self.y][self.x]: return True

            # splitter is fine
            return False
        
        def processTile(self):
            tile = input[self.y][self.x]
            energized[self.y][self.x] = True
            if tile == ".": return [self]
            if tile in "/\\":
                self.direction = dir_dicts[tile][self.direction]
                return [self]
            if tile in "-":
                if self.direction in "EW": return [self]
                self.direction = "E"
                return [self, Beam(self.x, self.y, "W")]
            if tile in "|":
                if self.direction in "NS": return [self]
                self.direction = "N"
                return [self, Beam(self.x, self.y, "S")]
            return Exception("What f***ing tile am I on?")
        
        def move(self):
            if self.direction == "N":
                self.y -= 1
            if self.direction == "E":
                self.x += 1
            if self.direction == "W":
                self.x -= 1
            if self.direction == "S":
                self.y += 1

    def launchBeam(start_beam=Beam(-1,0,"E")) -> int:

        # clear energized
        clearEnergized()


        beams = [start_beam]

        while beams:
            tmp_beams = beams
            beams = []
            for beam in tmp_beams:
                beam.move()
                if beam.checkEnd(): continue
                for new_beam in beam.processTile(): beams.append(new_beam)

        return countEnergized()

    if not ADVANCED:
        return launchBeam()

    max_energized = 0

    for start_x in range(WIDTH):
        local_max = max(launchBeam(Beam(start_x, -1, "S")), launchBeam(Beam(start_x, HEIGHT, "N")))
        max_energized = max(max_energized, local_max)

    for start_y in range(HEIGHT):
        local_max = max(launchBeam(Beam(-1, start_y, "E")), launchBeam(Beam(WIDTH, start_y, "W")))
        max_energized = max(max_energized, local_max)

    return max_energized

def daySeventeen(input):
    
    import networkx as nx

    WIDTH = len(input[0])
    HEIGHT = len(input)

    MIN_LINE = 4 if ADVANCED else 1
    MAX_LINE = 10 if ADVANCED else 3

    # create node graph
    edges = [
        ("start", "0,0,H", 0), ("start", "0,0,V", 0),
        (f"{WIDTH-1},{HEIGHT-1},H", "end", 0),
        (f"{WIDTH-1},{HEIGHT-1},V", "end", 0)
        ]

    def printPath(path):
        start_nodes = set([edge[:-2] for edge in path])
        print(start_nodes)
        for y in range(HEIGHT):
            line = ""
            for x in range(WIDTH):
                if f"{x},{y}" in start_nodes:
                    line+="#"
                else:
                    line+="."
            print(line)
                



    for y, row in enumerate(input):
        for x, _ in enumerate(row):
            for direction in "VH":
                curr_node = f"{x},{y},{direction}"
                new_direction = ""
                for offset in [-1,1]:
                    new_x = x
                    new_y = y
                    loss = 0
                    for dist in range(1,MAX_LINE+1):
                        if direction == "V":
                            new_y = y + dist*offset
                            new_direction = "H"
                        else:
                            new_x = x + dist*offset
                            new_direction = "V"
                        if new_x < 0 or new_x >= WIDTH: break
                        if new_y < 0 or new_y >= HEIGHT: break
                        loss += int(input[new_y][new_x])
                        if dist < MIN_LINE: continue
                        edges.append((curr_node, f"{new_x},{new_y},{new_direction}", loss))

    #all_nodes = ["start", "end"]
    #for y in HEIGHT:
    #    for x in WIDTH:
    #        all_nodes.append(f"{x},{y},H")
    #        all_nodes.append(f"{x},{y},V")

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)

    path = nx.algorithms.shortest_path(G,"start","end",weight="weight")
    for from_node, to_node in zip(path[:-1], path[1:], strict=True):
        print(from_node, to_node, G[from_node][to_node]['weight'])
    printPath(path[1:])
    return nx.algorithms.shortest_path_length(G, "start", "end", weight="weight")

def dayEighteen(input):

    if ADVANCED:
        
        # get vertices and boundary points
        commands = []
        for line in input:
            grps = re.match(r"([UDLR]) ([1-9][0-9]*) \(#([0-9a-f]{6})\)", line).groups()
            distance = int(grps[2][:5], 16)
            direction = "RDLU"[int(grps[2][5])]
            commands.append((distance, direction))

        boundary_count = 0
        x = 0
        y = 0
        vertices = [(x,y)]
        for distance, direction in commands:
            boundary_count += distance
            match direction:
                case "U": y -= distance
                case "D": y += distance
                case "L": x -= distance
                case "R": x += distance
            vertices.append((x,y))

        # find area (shoelace)
        area = 0
        for (x1, y1), (x2, y2) in zip(vertices[-2::-1], vertices[-1:0:-1], strict=True):
            area += x1*y2 - x2*y1
        area /= 2
        print(area)

        # unpick area
        interior = area + 1 - boundary_count/2

        return boundary_count + interior



    from math import inf

    directions = {
        "U": {"L": "\\", "R": "/"},
        "D": {"L": "/", "R": "\\"},
        "L": {"U": "\\", "D": "/"},
        "R": {"U": "/", "D": "\\"},
    }

    def findBoundaries(commands) -> (int, int, int, int):
        
        min_x = inf
        min_y = inf

        max_x = -inf
        max_y = -inf

        curr_x = 0
        curr_y = 0

        for direction, distance in commands:
            if direction == "U":
                curr_y -= distance
            if direction == "D":
                curr_y += distance
            if direction == "L":
                curr_x -= distance
            if direction == "R":
                curr_x += distance
            min_x = min(min_x, curr_x)
            min_y = min(min_y, curr_y)
            max_x = max(max_x, curr_x)
            max_y = max(max_y, curr_y)

        return (min_x, min_y, max_x, max_y)
    
    def processArray(arr, horiz_change) -> int:
        
        def processLine(line) -> int:
            is_inside = False
            acc = 0
            last_bend = ""
            last_x = 0
            s_line = sorted(line, key=lambda x: x[0])

            print_line = []

            for x, sym, dist in s_line:
                acc_diff = acc
                if is_inside: acc += x - (last_x)
                last_x = x + dist
                acc += dist
                inside_swap = False
                if sym in "/\\":
                    if not last_bend:
                        last_bend = sym
                    else:
                        if last_bend != sym:
                            last_bend = ""
                        else: inside_swap = True
                if sym == "|":
                    inside_swap = True

                if inside_swap:
                    is_inside = not is_inside
                print_line.append((x, sym, dist))
                print_line.append(acc-acc_diff)

            #print(print_line)
                


            return acc

        last_count = 0
        acc = 0
        for line, change_flag in zip(arr, horiz_change, strict=True):
            if change_flag:
                last_count = processLine(line)
            acc += last_count
        return sum([processLine(line) for line in arr])

    # process commands
    commands = []
    for line in input:
        grps = re.match(r"([UDLR]) ([1-9][0-9]*) \(#([0-9a-f]{6})\)", line).groups()
        commands.append((grps[0], int(grps[1])))
    
    min_x, min_y, max_x, max_y = findBoundaries(commands)

    arr = [[] for _ in range(max_y+1-min_y)]
    horiz_change = [False for _ in range(max_y+1-min_y)]

    x = -min_x
    y = -min_y

    start_dir = commands[0][0]

    for i, (direction, distance) in enumerate(commands):
        turn_direction = directions[direction][start_dir] if i == len(commands)-1 \
            else directions[direction][commands[i+1][0]]
        
        if direction in "UD":
                for ii in range(distance):
                    y += 1 if direction == "D" else -1
                    if ii == distance-1: break
                    arr[y].append((x, "|", 1))
                arr[y].append((x, turn_direction, 1))                   
        if direction == "L":
                x -= distance
                arr[y].append((x, turn_direction, 1))
                if distance > 1:
                    arr[y].append((x+1, "-", distance-1))
        if direction == "R":
                if distance > 1:
                    arr[y].append((x+1, "-", distance-1))
                x += distance
                arr[y].append((x, turn_direction, 1))
        if direction in "LR":
            horiz_change[y] = True
            
    return processArray(arr, horiz_change)

def dayNineteen(input):
    
    from math import prod

    class Step:

        def __init__(self, string):
            
            attr, cmp_sign, val, dest = re.match(
                r"([xmas])([<>])([0-9]+):(.*)", string
            ).groups()

            self.attr = attr
            self.is_gt = (cmp_sign==">")
            self.val = int(val)
            self.dest = dest

        def process(self, part) -> bool:
            in_val = part[self.attr]
            if in_val == self.val: return False
            return (in_val < self.val) ^ self.is_gt
        
        def processComposite(self, part) -> (bool, bool):
            min_attr, max_attr = part.attrs[self.attr]
            return (
                (min_attr < self.val) ^ self.is_gt,
                (max_attr < self.val) ^ self.is_gt
            )


    class Workflow:

        def __init__(self, string):
            steps = string.split(',')
            self.steps = [Step(stepstr) for stepstr in steps[:-1]]
            self.final = steps[-1]

    def newWorkflow(workflow_line) -> (str, Workflow):
        wf_name, wf_desc = re.match(r"([a-z]+)\{(.+)\}", workflow_line).groups()
        return wf_name, Workflow(wf_desc)

    class CompositePart:


        def __init__(self, max=4000):
            self.attrs = {}
            for attr in "xmas":
                self.attrs[attr] = (1, max)

        def changeRange(self, attr, min_attr, max_attr):
            self.attrs[attr] = (min_attr, max_attr)

        def getCombinations(self) -> int:
            return prod([max_attr - min_attr + 1 for min_attr, max_attr in self.attrs.values()])

        def __repr__(self):
            return str(self.attrs)

    def copy(part: CompositePart) -> CompositePart:
        new_part = CompositePart()
        for attr in "xmas":
            new_part.changeRange(attr, *part.attrs[attr])
        return new_part
    

    workflows = {}
    
    def resolvePart(part, wf_name):
        if wf_name == "A":
            return sum(part.values())
        if wf_name == "R":
            return 0
        
        wf = workflows[wf_name]

        for step in wf.steps:
            if step.process(part):
                return resolvePart(part, step.dest)
        return resolvePart(part, wf.final)
    
    def resolveCompositePart(part: CompositePart, wf_name: str) -> int:
        if wf_name == "R":
            return 0
        if wf_name == "A":
            return part.getCombinations()
        
        wf = workflows[wf_name]

        acc = 0

        for step in wf.steps:
            attr = step.attr
            min_attr, max_attr = part.attrs[attr]
            low_pass, high_pass = step.processComposite(part)
            if low_pass == high_pass:
                if low_pass:
                    return acc + resolveCompositePart(part, step.dest)
            else:
                break_point = step.val if step.is_gt else (step.val - 1)
                new_part = copy(part)
                if not low_pass:
                    part.changeRange(attr, min_attr, break_point)
                    new_part.changeRange(attr, break_point+1, max_attr)
                else:
                    part.changeRange(attr, break_point+1, max_attr)
                    new_part.changeRange(attr, min_attr, break_point)
                acc += resolveCompositePart(new_part, step.dest)
        return acc + resolveCompositePart(part, wf.final)

        

    workflows_flag = True
    acc = 0
    for line in input:
        
        if workflows_flag:
            if not line:
                workflows_flag = False
                continue
            wf_name, wf = newWorkflow(line)
            workflows[wf_name] = wf
            continue

        if ADVANCED:

            return resolveCompositePart(CompositePart(), "in")
        
        part = {
            attr: int(val) for attr, val in zip("xmas",
                re.match(r"\{x=([0-9]+),m=([0-9]+),a=([0-9]+),s=([0-9]+)\}", line).groups()
            )
        }
        acc += resolvePart(part, "in")
    
    return acc

def dayTwenty(input):

    # pulse definition:
    # (source, is_hi, dest)

    class Part:

        def __init__(self):
            ...
        
        def receive(self, pulse: (str, bool, str)) -> [(str, bool, str)]:
            return []

    class Broadcaster(Part):
        
        def __init__(self, dest_list: [str]):
            self.dest_list = dest_list

        def receive(self, pulse: (str, bool, str)) -> [(str, bool, str)]:
            _, is_hi, my_name = pulse
            return [(my_name, is_hi, dest) for dest in self.dest_list]
        
    class FlipFlop(Part):

        def __init__(self, dest_list):
            self.dest_list = dest_list
            self.is_on = False

        def receive(self, pulse: (str, bool, str)) -> [(str, bool, str)]:
            _, is_hi, my_name = pulse
            if is_hi: return []
            self.is_on = not self.is_on
            return [(my_name, self.is_on, dest) for dest in self.dest_list]
        
    class Conjunction(Part):

        def __init__(self, dest_list: [str], src_list: [str]):
            self.dest_list = dest_list
            self.last_input = {}
            for src in src_list:
                if src == "roadcaster":
                    self.last_input["broadcaster"] = False
                else:
                    self.last_input[src] = False

        def receive(self, pulse: (str, bool, str)) -> [(str, bool, str)]:
            src, is_hi, my_name = pulse
            self.last_input[src] = is_hi
            return [(my_name, not all(self.last_input.values()), dest) for dest in self.dest_list]

    class Network:

        def __init__(self, conn_list: [(str, [str])]):
            self.parts = {}
            self.his = 0
            self.los = 0

            self.rx_got_low = False

            for name, dest_list in conn_list:
                match name[0]:
                    case "b": self.parts["broadcaster"] = Broadcaster(dest_list)
                    case "%": self.parts[name[1:]] = FlipFlop(dest_list)
                    case "&": self.parts[name[1:]] = Conjunction(dest_list,
                                                                    [src[1:] for src, dests in conn_list if name[1:] in dests]
                                                                 )
                for dest in dest_list:
                    if dest not in self.parts.keys(): self.parts[dest] = Part()


        def push(self) -> bool:
            signal_q = [("button", False, "broadcaster")]
            while signal_q:
                src, is_hi, dest = signal_q.pop(0)
                #print(src, "-high->" if is_hi else "-low->", dest)
                if not is_hi and dest == "rx": self.rx_got_low = True
                if is_hi:
                    self.his += 1
                else:
                    self.los += 1
                try:
                    signal_q.extend(self.parts[dest].receive((src, is_hi, dest)))
                except:
                    print(self.parts)
                    raise
            return self.rx_got_low

        def calculate(self) -> int:
            return self.his * self.los

    # prepare input
    conn_list = []
    for line in input:
        src, dests = line.split(" -> ")
        conn_list.append((src, dests.split(", ")))

    network = Network(conn_list)

    if ADVANCED:
        pushes = 0
        while not network.rx_got_low:
            pushes += 1
            network.push()
            if any(network.parts["qb"].last_input.values()):
                print(pushes, network.parts["qb"].last_input)

        return pushes

    for i in range(1000):
        network.push()

    return network.calculate()


scripts = [
    dayOne, dayTwo, dayThree, dayFour, dayFive,
    daySix, daySeven, dayEight, dayNine, dayTen,
    dayEleven, dayTwelve, dayThirteen, dayFourteen,
    dayFifteen, daySixteen, daySeventeen, dayEighteen,
    dayNineteen, dayTwenty
]

def getInputFromNumber(number, test=False):
    fname = os.path.join(".", "inputs_test" if test else "inputs", f"day{number}.txt")
    with open(fname, "r") as f:
        output = f.read().splitlines()
    return output

def doTask(date, task_input):
    return scripts[date-1](task_input)

if __name__=="__main__":
    
    ap = argparse.ArgumentParser(
        prog='aoc2023solver',
        description='general script for all AOC2023 problems',
        epilog='hello world!'
    )

    ap.add_argument('num', type=int)
    ap.add_argument('-t', '--test', action='store_true')
    ap.add_argument('-a', '--advanced', action='store_true')

    args = ap.parse_args()

    date = args.num
    if date < 1 or date > 25:
        raise ValueError("AOC date must be between 1 and 25.")
        
    task_input = getInputFromNumber(date, args.test)

    ADVANCED = args.advanced

    output = doTask(date, task_input)

    print(output)

    if OUTPUT:
        with open("out.txt", "w+") as f:
            f.write('\n'.join(OUTPUT))