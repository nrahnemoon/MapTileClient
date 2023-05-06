"""
This script updates the README.md file by reading its contents, adding the new entry
based on the increment type, version, and description, and writing the updated release notes
back to the file.

Usage:
    python update_release_notes.py <version> <description>

Example:
    python update_release_notes.py minor "0.1.0" "Added a new feature"
"""

import sys
import datetime

def update_release_notes(version, description):
    with open('README.md', 'r') as f:
        readme_contents = f.readlines()

    # Get current date in the required format
    current_date = datetime.date.today().strftime('%Y-%m-%d')

    # Construct the new release note line
    release_note = f"\n*{version}* ({current_date}) {description}\n"

    # Find the index of the "## Release Notes" line
    release_notes_start_index = None
    for i, line in enumerate(readme_contents):
        if line.strip() == "## Release Notes":
            release_notes_start_index = i
            break

    if release_notes_start_index is None:
        print("Error: Couldn't find the 'Release Notes' section in the README.md file.")
        sys.exit(1)

    # Find the index of the last release note in the "Release Notes" section
    last_release_note_index = None
    for i in range(release_notes_start_index + 1, len(readme_contents)):
        if readme_contents[i].startswith("*"):
            last_release_note_index = i
        elif readme_contents[i].startswith("##") or i == len(readme_contents) - 1:
            if last_release_note_index is None:
                last_release_note_index = i
            break

    if last_release_note_index is None:
        print("Error: Couldn't find any release notes in the 'Release Notes' section.")
        sys.exit(1)

    # Add the new release note line after the index of the last release note
    readme_contents.insert(last_release_note_index + 1, release_note)

    # Write the updated contents back to the README.md file
    with open('README.md', 'w') as f:
        f.writelines(readme_contents)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update_release_notes.py <VERSION> <CHANGELOG_DESCRIPTION>")
        sys.exit(1)

    update_release_notes(sys.argv[1], sys.argv[2])
