# Urban-Dev

This project involves web scraping, data processing, and machine learning model training. It consists of scripts to extract data from web sources, process the extracted information, and train a predictive model.

## Folder Structure
```
.
├── data.txt               # Historical dataset with timestamps and numerical values
├── loaction.txt           # Mapping of locations to web URLs
├── webScript.txt          # JavaScript for extracting web data
├── dataScrap.py           # Python script for web scraping
├── mainScript.py          # Deep learning model training
├── scrapRunner.py         # Runner script for dataScrap.py
├── trainingPipeline.py    # Model training pipeline using TensorFlow
```

## Requirements
To run the scripts, ensure you have the following dependencies installed:
```bash
pip install selenium tqdm pillow numpy tensorflow scikit-learn matplotlib imageio wandb
```
Additionally, you need:
- A web driver for Selenium (e.g., GeckoDriver for Firefox)
- A valid dataset and download path configured in `dataScrap.py`

## Scripts
### Web Scraping
- **`webScript.txt`**: A JavaScript snippet to extract date and release numbers from web elements.
- **`dataScrap.py`**: Automates web scraping using Selenium, saves screenshots, and processes extracted data.
- **`scrapRunner.py`**: Loads historical data and executes `dataScrap.py` to collect images for different timestamps.

### Machine Learning Model
- **`mainScript.py`**: Implements a deep learning model using CNN-LSTMs to process sequential image data.
- **`trainingPipeline.py`**: Similar to `mainScript.py` but integrates Weights & Biases (`wandb`) for experiment tracking.

## Usage
### Running Web Scraping
```bash
python scrapRunner.py
```
Ensure `data.txt` is populated with valid timestamps and release numbers.

### Training the Model
```bash
python mainScript.py
```
Ensure `dataset_path` is correctly set to the image directory.

## Output
- Web scraping outputs images stored in the specified folder.
- Model training saves the trained model as `Model.h5` and generates predictions.

## Author
This project was developed by **Nikita Kodkany**.

