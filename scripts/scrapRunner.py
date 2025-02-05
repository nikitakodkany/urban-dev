import time
import glob
import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image

# Configuration
MAP_COORDS = "-118.33681%2C34.08500%2C15"
USER = ""
LOCATION = "UCLA"
FILE_PATH = "data.txt"  # Replace with actual path
DOWNLOAD_PATH = f'C:/Users/{USER}/Downloads/{LOCATION}'

# Ensure download directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Function to read data from text file
def load_data(file_path):
    data = {}
    with open(file_path, "r") as file:
        for line in file:
            date, number = line.strip().split()
            data[date] = int(number)
    return data

# Function to initialize WebDriver
def init_browser():
    browser = webdriver.Firefox()
    browser.maximize_window()
    return browser

# Function to capture screenshot
def capture_screenshot(browser, date, num):
    link = f'https://livingatlas.arcgis.com/wayback/#active={num}&mapCenter={MAP_COORDS}'
    browser.get(link)
    time.sleep(2)
    
    # Accept cookies
    try:
        browser.find_element(By.ID, "onetrust-accept-btn-handler").click()
        time.sleep(2)
    except Exception:
        pass
    
    # Click checkbox
    try:
        browser.find_element(By.CSS_SELECTOR, "div.margin-left-half.margin-right-quarter.cursor-pointer").click()
        time.sleep(2)
    except Exception:
        pass
    
    # Save and crop screenshot
    screenshot_path = os.path.join(DOWNLOAD_PATH, 'temp.png')
    browser.save_screenshot(screenshot_path)
    
    crop_and_save_image(screenshot_path, date)

# Function to crop and save image
def crop_and_save_image(image_path, date):
    left, upper, right, lower = 400, 60, 1910, 930
    img = Image.open(image_path).crop((left, upper, right, lower))
    img.save(os.path.join(DOWNLOAD_PATH, f'{date}.png'))

# Main Execution
def main():
    data_dict = load_data(FILE_PATH)
    
    for date, num in tqdm(data_dict.items(), desc="Capturing Screenshots"):
        browser = init_browser()
        capture_screenshot(browser, date, num)
        browser.quit()

if __name__ == "__main__":
    main()
