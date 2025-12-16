#!/bin/bash
# Setup git repository for p2cs_project

cd /zdata/user-data/noam/p2cs_project

# Initialize git repository
git init

# Add remote (update with your GitHub username if different)
git remote add origin https://github.com/segal-noam/p2cs_project.git

# Add all files (data/ folder will be ignored by .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: P2CS project"

# Rename branch to main
git branch -M main

echo "Git repository initialized!"
echo "Next steps:"
echo "1. Create the repository 'p2cs_project' on GitHub if you haven't already"
echo "2. Set up authentication (SSH or personal access token)"
echo "3. Run: git push -u origin main"

