# Protein-Explorer-Atlas
App that provides intelligent consultations on protein-related topics using natural language processing (NLP) for users who may not have access to large-scale AI resources. The goal is to democratize scientific research and empower users with knowledge about the role of in various biological processes.

python env:
python venv 

## 2. Create and Activate a Virtual Environment

It's best practice to use a virtual environment to keep project dependencies isolated.

Create the virtual environment:
```sh
python3 -m venv HPA_Explorer
```

Now, activate it. The command varies based on your operating system:

### For macOS and Linux:
```sh
source HPA_Explorer/bin/activate
```

### For Windows:
In Command Prompt:
```cmd
HPA_Explorer\Scripts\activate.bat
```
In PowerShell:
```powershell
HPA_Explorer\Scripts\Activate.ps1
```

## 3. Install Dependencies

With the virtual environment active, install the necessary packages:
```sh
pip install -r requirements.txt
```

## 4. Run the Project

You can now launch the application:
```sh
python main.py
```