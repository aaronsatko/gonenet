import os

def print_directory_structure(path, indent=""):
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                print(f"{indent}├── {entry.name}/")
                print_directory_structure(entry.path, indent + "│   ")
            else:
                print(f"{indent}├── {entry.name}")

if __name__ == "__main__":
    root_dir = "C:\\Users\\agsat\\Desktop\\gonenet"
    print(f"{root_dir}/")
    print_directory_structure(root_dir, "    ")