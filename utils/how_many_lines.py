import os
import time


# https://successfulsoftware.net/2017/02/10/how-much-code-can-a-coder-code/
# https://en.wikipedia.org/wiki/Capers_Jones
# Capers Jones measured productivity of around 16 to 38 LOC per day across a range of projects.


def count_lines_of_code_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)  # if line.strip())


def count_lines_of_code_in_directory(directory):
    total_lines = 0
    total_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_files += 1
                total_lines += count_lines_of_code_in_file(file_path)
    return total_lines, total_files


def main():
    current_directory = os.getcwd()
    total_lines, total_files = count_lines_of_code_in_directory(current_directory)
    print(f"Total lines of code in {total_files} python files: {total_lines}")
    # Days off: 2023: 8, 2024: 6, Vacation: 22, Conferences: 19
    #print(f"Lines of code per day: {total_lines/(325-14-22-19):.2f}")


if __name__ == "__main__":
    main()
