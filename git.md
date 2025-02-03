I'll help you switch from your work Git to personal Git credentials. Here's the step-by-step process:

1. First, logout from your work Git:
```bash
git config --global --unset user.name
git config --global --unset user.email
git config --global --unset credential.helper
```

2. Clear any cached credentials using:
```bash
git credential-manager delete https://github.com
```

3. Configure your personal Git credentials:
```bash
git config --global user.name "your-personal-username"
git config --global user.email "your-personal-email"
```

4. For your personal project, navigate to its directory and set up the repository:
```bash
cd path/to/your/project
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```