from typing import Dict, List, Tuple

def get_block_info_lines(file_path: str) -> List[str]:
    input = open(file_path, "r")
    lines = [line for line in input]
    input.close()
    info_lines: List[str] = []
        
    for i, line in enumerate(lines):
        if "<=" not in line:
            continue
        if i + 1 < len(lines) and "<=" in lines[i + 1]:
            continue
        if i + 1 >= len(lines):
            break
        if "stdin" in lines[i + 1]:
            continue
        info_lines.append(
            (lines[i].split("<=")[0].strip(), lines[i + 1])
        )
    return info_lines
    

def get_current(line: str) -> str:
    return line.split("+")[1].strip().split(" ")[0]
    
def get_jumped(line: str) -> str:
    return line.split("+")[2].strip().split(" ")[0]
    
def get_non_jumped(line: str) -> str:
    return line.split("+")[3].strip().split(" ")[0]
    
def get_details(file_path: str) -> List[Tuple[str, str, str, str]]:
    return [
        (line[0], get_current(line[1]), get_jumped(line[1]), get_non_jumped(line[1]))
        for line in get_block_info_lines(file_path)
    ]

#print(get_details("trace_examples/example.txt"))

#
details: List[Tuple[str, str, str, str]] = get_details("trace_examples/example.txt")
groups: Dict[str, List[str]] = {}
for i, entry in enumerate(details):
    if i - 1 >= 0 and details[i][1] == details[i - 1][2]:
        current: str = details[i - 1][0] 
        jumped: str = details[i][0]
        if current not in groups:
            groups[current] = [jumped]
        else:
            groups[current].append(jumped)
        
print("Conected components with more than one nodes: ")
for key, value in groups.items():
    print(f"{key}, {', '.join(value)}")

