## Application Usage - Local Use

This folder includes setup instructions and usage examples for the application. Follow the steps below to install the dependencies and run the app locally.

---

## Run the App Locally

The following steps will demonstrate how to set up our virtual environment, set up, and run the application on your local machine.  
Prerequisites: you will need a command line interface with Conda installed.

### Steps for conda download

This project requires **Conda** to manage the virtual environment and dependencies. Follow the steps below to install Conda on your system if you haven't done so already.

---

#### 1. Download Miniconda

We recommend installing **Miniconda**, which is a lightweight version of Anaconda.

Download Miniconda from the official website:

https://docs.conda.io/en/latest/miniconda.html

Choose the installer that matches your operating system:

- **Mac (Intel or Apple Silicon)**
- **Windows**
- **Linux**

---

#### 2. Install Miniconda

Run the downloaded installer and follow the installation prompts.

For Mac or Linux, installation typically looks like:

```bash
bash Miniconda3-latest-MacOSX-*.sh
```

---

## 1. Clone the Git Repository

To obtain a local copy of the repository, run the following command:

```bash
git clone https://github.com/allison2368/Netflix-User-Behavior
cd Netflix-User-Behavior
```

## 2. Local Environment Setup

Create the Conda environment netflix-env using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

Once the environment is created, activate it using:

```bash
conda activate netflix-env
```

## 3. Configure Google Cloud

Install Google Cloud SDK if you don't have it already

Install via Homebrew.

```bash
brew install --cask google-cloud-sdk
```

After installing, restart your terminal then run:

```bash
gcloud init
```

Log in and configure credentials. This opens a browser, you log in, and ADC credentials are stored locally for the Python client libraries.

```bash
gcloud auth application-default login
```

## 4. Run app locally

You need to configure the project id before running the app
```bash
export PROJECT_ID='netflix-user-behavior'
```
```bash
streamlit run dashboard.py
```
