# ANITA-Anti-Inflammatory-Peptide-Predictor

This tool is fully containerized using **Docker**.

## Prerequisites

You don't need to install Python, complex machine learning libraries, or resolve dependency conflicts. The only requirement is Docker.

* **Install Docker Desktop:**
    * **Windows / Mac:** Download and install from the [official Docker website](https://www.docker.com/products/docker-desktop/).
* Make sure the Docker application is running in the background before proceeding.

---

## Quick Start

The easiest way to use the AIP-Predictor is to pull the pre-built Docker image directly from Docker Hub.

### 1. Prepare your input data
Create a folder on your computer and place your input `.fasta` file inside it. 

### 2. Run the prediction
Open your terminal (Command Prompt, PowerShell, or bash) and run the following command. 

```bash
docker run --rm –v "YOUR_FOLDER:/app/data/output" takoyakiyee/aip-predictor:latest python main_Predict.py --input /app/data/output/your fasta name
```
If you have more than two FASTA files:
```bash
docker run --rm –v "YOUR_FOLDER:/app/data/output" takoyakiyee/aip-predictor:latest python main_Predict.py --input /app/data/output/YOUR_FIRST_FASTA_NAME.fasta /app/data/output/YOUR_SECOND_FASTA_NAME.fasta
```
### 2.1 Note:
 1. Replace 🔴 **`YOUR_FOLDER`** with the actual absolute path to your local folder. 
 2. Replace 🔴 **`YOUR_FASTA_NAME.fasta`** with the exact name of your FASTA file.
 3. Replace 🔴 **`YOUR_FIRST_FASTA_NAME.fasta`** and 🔴 **`YOUR_SECOND_FASTA_NAME.fasta`** with the exact name of your FASTA file.

### 📂 Output Files
predVector.csv -- The prediction output in binary format (1 for positive and 0 for negative).

probVector.csv -- The prediction probability estimate.

### Docker Hub
Website: https://hub.docker.com/r/takoyakiyee/aip-predictor
