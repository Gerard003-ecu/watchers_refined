# format_script.py
import os

file_path = "watchers/watchers_tools/malla_watcher/malla_watcher.py"
original_content = ""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        original_content = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

modified_content = original_content

# Specific fix for F841 at/around original line 285
# Handle potential variations due to previous formatting attempts
modified_content = modified_content.replace(
    "except Exception as e:  # F841: e was unused",
    "except Exception:  # F841: e was unused"
)
modified_content = modified_content.replace(
    "except Exception as e:",
    "except Exception:"
)

# Specific fix for F824 at/around original line 379
modified_content = modified_content.replace(
    "global aggregate_state  # F824: Unused global declaration",
    "# global aggregate_state  # F824: Unused global declaration"
)
modified_content = modified_content.replace(
    "global aggregate_state",
    "# global aggregate_state"
)

# Apply yapf formatting
try:
    from yapf.yapflib import yapf_api
    # Style based on PEP8, with column limit 79
    style_config = "{based_on_style: pep8, column_limit: 79}"
    modified_content, changed = yapf_api.FormatCode(
        modified_content,
        filename=file_path,
        style_config=style_config,
        print_diff=False
    )
    print(f"Yapf applied. Changed: {changed}")
except Exception as e:
    print(f"Error during yapf formatting: {e}")
    # Proceed with content as modified so far, yapf might not be critical if other fixes work
    pass # Or exit(1) if yapf is essential

try:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    print(f"Successfully wrote formatted content to {file_path}")
except Exception as e:
    print(f"Error writing file: {e}")
    exit(1)
