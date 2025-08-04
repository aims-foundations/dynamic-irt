# Contributing Guidelines

Thank you for your interest in contributing to **[PROJECT\_NAME]**! To ensure a smooth process, please follow these guidelines.

## 1. Code of Conduct

Please read and abide by our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## 2. Signing the CLA

Before submitting any pull request, you must sign our Contributor License Agreement (CLA):

1. Open and fill out the `CONTRIBUTOR_LICENSE_AGREEMENT.md` in the project root.
2. Commit your completed CLA file to your branch.
3. Ensure it’s included in your pull request.

Any pull requests without a signed CLA will not be accepted.

## 3. Getting Started

1. **Fork** the repository and **clone** your fork:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/[PROJECT_REPOSITORY_NAME].git
   ```
2. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Install dependencies** and setup:
   ```bash
   # example
   pip install -r requirements.txt
   ```
4. **Run tests** to ensure everything is passing:
   ```bash
   pytest
   ```

## 4. Contribution Workflow

- **Issue**: Create an issue before implementing a large feature to discuss design and scope.
- **Branch**: Name branches clearly, e.g., `bugfix/login-crash` or `feat/add-qlma-metric`.
- **Commits**: Write clear, descriptive commit messages. Use the imperative mood (e.g., “Add feature”, not “Added feature”).
- **Pull Request**:
  - Reference the issue by number (e.g., “Fixes #123”).
  - Include a summary of changes and rationale.
  - Ensure all checks (CI/tests) pass.

## 5. Style and Testing

- Follow the existing coding style (PEP8 for Python, etc.).
- Add or update tests for new features or bug fixes.
- Run linting before committing:
  ```bash
  flake8 .
  ```

## 6. Documentation

- Update `README.md` or other documentation for any new configuration, commands, or features.
- Ensure examples in docs remain accurate.

## 7. License and Copyright

By contributing, you agree that your contributions will be licensed under the project’s license. Consult the `LICENSE` file for details.

---

Thank you for making **[CodeInsights]** better! We appreciate your time and effort. Feel free to reach out with any questions or feedback.

