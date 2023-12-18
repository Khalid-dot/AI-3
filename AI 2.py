import subprocess
import sys

def install_libraries():
    try:
        import numpy
        import pygame
    except ImportError:
        print("Required libraries not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pygame"])

# Install required libraries
install_libraries()

# Now you can proceed with the rest of your code
import numpy as np
import heapq
import pygame
import sys

# ... (rest of your code)


# Constants for visualization
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

import numpy as np
import random

def generate_maze(enrollment_number):
    last_two_digits = int(enrollment_number[-2:])
    maze_size = last_two_digits
    maze = np.zeros((maze_size, maze_size), dtype=int)

    # Set start and goal positions
    start_position = (0, 0)
    goal_position = (maze_size - 1, maze_size - 1)

    # Randomly place obstacles
    obstacle_density = 0.2  # Adjust this value to control obstacle density
    for row in range(maze_size):
        for col in range(maze_size):
            if (row, col) not in [start_position, goal_position] and random.random() < obstacle_density:
                maze[row, col] = 1  # Set obstacle

    return maze, start_position, goal_position


def heuristic(position, goal):
    return np.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)

def astar_search(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break

        for neighbor in neighbors(current_node, maze):
            new_cost = cost_so_far[current_node] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    return came_from, cost_so_far


def dijkstra_search(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break

        for neighbor in neighbors(current_node, maze):
            new_cost = cost_so_far[current_node] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost, neighbor))
                came_from[neighbor] = current_node

    return came_from, cost_so_far

def neighbors(node, maze):
    row, col = node
    adjacent_nodes = []

    # Check UP
    if row - 1 >= 0 and maze[row - 1, col] == 0:
        adjacent_nodes.append((row - 1, col))

    # Check DOWN
    if row + 1 < maze.shape[0] and maze[row + 1, col] == 0:
        adjacent_nodes.append((row + 1, col))

    # Check LEFT
    if col - 1 >= 0 and maze[row, col - 1] == 0:
        adjacent_nodes.append((row, col - 1))

    # Check RIGHT
    if col + 1 < maze.shape[1] and maze[row, col + 1] == 0:
        adjacent_nodes.append((row, col + 1))

    return adjacent_nodes

def extract_path(came_from, start, goal):
    current_node = goal
    path = []

    while current_node != start:
        path.append(current_node)
        try:
            current_node = came_from[current_node]
        except KeyError:
            print("Goal not reached.")
            return []

    # Add the start node to the path
    path.append(start)

    # Reverse the path to get it from start to goal
    path.reverse()

    return path

def convert_to_4p_moves(path):
    moves = []
    for i in range(1, len(path)):
        current_node = path[i - 1]
        next_node = path[i]
        row_diff = next_node[0] - current_node[0]
        col_diff = next_node[1] - current_node[1]

        if row_diff == 1:
            moves.append('DOWN')
        elif row_diff == -1:
            moves.append('UP')
        elif col_diff == 1:
            moves.append('RIGHT')
        elif col_diff == -1:
            moves.append('LEFT')

    return moves

def draw_maze(screen, maze, start_position, goal_position, CELL_SIZE, enrollment_number):
    font = pygame.font.Font(None, 36)
    # Display enrollment number at the top
    enrollment_text = font.render(f"Enrollment: {enrollment_number}", True, (255, 255, 255))
    screen.blit(enrollment_text, (10, 10))

    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (col * CELL_SIZE, (row + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            if (row, col) == start_position:
                pygame.draw.rect(screen, (0, 255, 0), (col * CELL_SIZE, (row + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                text = font.render('S', True, (0, 0, 0))
                screen.blit(text, (col * CELL_SIZE + CELL_SIZE // 3, (row + 1) * CELL_SIZE + CELL_SIZE // 3))
            elif (row, col) == goal_position:
                pygame.draw.rect(screen, (255, 0, 0), (col * CELL_SIZE, (row + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                text = font.render('G', True, (0, 0, 0))
                screen.blit(text, (col * CELL_SIZE + CELL_SIZE // 3, (row + 1) * CELL_SIZE + CELL_SIZE // 3))

def draw_path(screen, path, CELL_SIZE, cost_so_far):
    font = pygame.font.Font(None, 36)

    if path:
        # Print path cost in the terminal
        print(f"Path Cost: {cost_so_far[path[-1]]}")

        for node in path[1:-1]:  # Exclude start and goal from the path
            pygame.draw.rect(screen, (0, 0, 255), (node[1] * CELL_SIZE, (node[0] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            text = font.render('X', True, (255, 255, 255))
            screen.blit(text, (node[1] * CELL_SIZE + CELL_SIZE // 3, (node[0] + 1) * CELL_SIZE + CELL_SIZE // 3))
            pygame.display.flip()
            pygame.time.wait(100)
    else:
        print("No path found.")


def main():
    enrollment_number = input("Enter your enrollment number: ")
    last_two_digits_sum = sum(int(digit) for digit in enrollment_number[-2:])
    maze_size = last_two_digits_sum
    print(f"Generated maze size: {maze_size}x{maze_size}")

    maze, start_position, goal_position = generate_maze(enrollment_number)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 50))  # Increased height for enrollment display
    pygame.display.set_caption("Maze Visualization")

    # Adjust CELL_SIZE based on maze size
    CELL_SIZE = SCREEN_WIDTH // maze.shape[0]

    # Draw initial maze
    draw_maze(screen, maze, start_position, goal_position, CELL_SIZE, enrollment_number)
    pygame.display.flip()

    # Wait for a key press to start the search
    input("Press Enter to start the search...")

    algorithm_choice = input("Enter D for Dijkstra or A for A* Algorithm: ").upper()

    if algorithm_choice == 'D':
        came_from, cost_so_far = dijkstra_search(maze, start_position, goal_position)
        path = extract_path(came_from, start_position, goal_position)
        moves = convert_to_4p_moves(path)
        print("Shortest path (Dijkstra):", moves)
        draw_path(screen, path, CELL_SIZE, cost_so_far)
    elif algorithm_choice == 'A':
        came_from, cost_so_far = astar_search(maze, start_position, goal_position)
        path = extract_path(came_from, start_position, goal_position)
        moves = convert_to_4p_moves(path)
        print("Shortest path (A*):", moves)
        draw_path(screen, path, CELL_SIZE, cost_so_far)
    else:
        print("Invalid algorithm choice. Please enter D or A.")

    # Wait for a key press to close the window
    input("Press Enter to exit...")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
