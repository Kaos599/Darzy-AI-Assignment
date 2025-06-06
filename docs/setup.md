# Setup and Installation Guide

This guide provides step-by-step instructions to set up and install the AI Fashion Analysis Tool on your local machine.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python:** Version 3.8 or newer. You can download Python from [python.org](https://www.python.org/downloads/).
*   **pip:** Python package installer. This is usually included with Python installations. You can check by running `pip --version`.
*   **git (Optional):** Required if you want to clone the repository. If you download the source code as a ZIP file, git is not strictly necessary. You can get git from [git-scm.com](https://git-scm.com/downloads).

## 2. Obtain the Source Code

You have two options to get the source code:

*   **Option A: Clone the Repository (Recommended)**
    Open your terminal or command prompt and run the following command:
    ```bash
    git clone <repository_url>
    cd aifashion  # Navigate into the project's root directory
    ```
    (Replace `<repository_url>` with the actual URL of the project's git repository.)

*   **Option B: Download the Source Code**
    If you prefer not to use git, you can download the source code (e.g., as a ZIP file) from its source (e.g., GitHub). Extract the archive into a directory on your computer. Ensure the final structure has `main.py`, `requirements.txt`, the `src/` folder, and the `tests/` folder directly within your chosen project directory (let's call it `aifashion/`).

## 3. Create and Activate a Virtual Environment (Highly Recommended)

Using a virtual environment is crucial for managing project dependencies and avoiding conflicts with other Python projects or your global Python installation.

Navigate to the project's root directory (`aifashion/`) in your terminal and run the following commands:

*   **Create the virtual environment:**
    ```bash
    python -m venv venv
    ```
    This creates a directory named `venv` within your project folder.

*   **Activate the virtual environment:**
    *   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows (Command Prompt/PowerShell):**
        ```bash
        venv\Scripts\activate
        ```
    After activation, your terminal prompt should change to indicate that you are now working within the `(venv)` environment.

## 4. Install Dependencies

With your virtual environment activated, install the required Python packages. The project includes a `requirements.txt` file that lists all necessary dependencies.

From the project root directory (`aifashion/`), run:
```bash
pip install -r requirements.txt
```
This command will download and install libraries such as Streamlit, Pillow, Langchain, and the Google Generative AI SDK.

## 5. Configure Environment Variables (API Key)

The application requires a Google Gemini API key to interact with the AI model for fashion analysis.

*   **Create a `.env` file:**
    In the root directory of the project (`aifashion/`), create a new file named `.env`.

*   **Add your API Key:**
    Open the `.env` file with a text editor and add your Gemini API key in the following format:
    ```env
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
    ```
    Replace `"YOUR_ACTUAL_GEMINI_API_KEY_HERE"` with your real API key.

*   **(Optional) Specify a Model Name:**
    You can also specify a particular Gemini vision model by adding another line to your `.env` file:
    ```env
    GEMINI_MODEL_NAME_VISION="gemini-pro-vision" # Or your preferred compatible model
    ```
    If this is not set, the application defaults to "gemini-pro-vision".

The application uses the `python-dotenv` library to automatically load these variables when it starts. If the `GEMINI_API_KEY` is not found in the environment or the `.env` file, the application will provide a text input field to enter it manually when you run the app.

---

You have now completed the setup process. You are ready to run the application and its unit tests. Refer to the [Usage Guide (`usage.md`)](usage.md) for instructions on how to start the application and use its features.
