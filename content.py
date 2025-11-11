import os

def write_python_tree_with_code(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            level = dirpath.replace(root_dir, "").count(os.sep)
            indent = "    " * level
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    f.write(f"{indent}    {filename}\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as code_file:
                            for line in code_file:
                                f.write(f"{indent}        {line.rstrip()}\n")
                    except Exception as e:
                        f.write(f"{indent}        [Error reading file: {e}]\n")

# Example usage:
write_python_tree_with_code("energysim/", "output.txt")
