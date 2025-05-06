import os
import re

def delete_ckpt_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "last.ckpt":
                # Delete last.ckpt
                file_path = os.path.join(subdir, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)
            else:
                # Check for files with the pattern epoch=<number>-val_score=<number>.ckpt
                match = re.match(r"epoch=\d+-val_score=([\d.]+)\.ckpt", file)
                if match:
                    val_score = float(match.group(1))
                    if val_score <= 40:
                        # Delete ckpt if validation score is less than or equal to 40
                        file_path = os.path.join(subdir, file)
                        print(f"Deleting: {file_path}")
                        os.remove(file_path)

if __name__ == "__main__":
    directory_path = "outputs"  # Change this if needed
    delete_ckpt_files(directory_path)
