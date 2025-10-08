import os
import sys

def directory_enumerator(dir_path: str, extensions: list[str], limit:int=None) -> list[str]:
    
    matching_files = []
    total = limit if limit is not None else "unlimited"

    for file in os.listdir(dir_path):
        if any(file.endswith(ext) for ext in extensions):
            full_path = os.path.join(dir_path, file)
            matching_files.append(full_path)

            count = len(matching_files)
            try:
                sys.stdout.write(f"\rFound: {count}/{total} -> {full_path}")
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Handle Unicode characters in file paths
                safe_path = full_path.encode('ascii', 'replace').decode('ascii')
                sys.stdout.write(f"\rFound: {count}/{total} -> {safe_path}")
                sys.stdout.flush()

            if limit is not None and count >= limit:
                break

    print()
    return matching_files
