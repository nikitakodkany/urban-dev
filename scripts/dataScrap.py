import time
import os
import sys
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image

# Read command-line arguments
mapCord, user, loc = sys.argv[1], sys.argv[2], sys.argv[3]

# Define file paths
file_path = "data.txt"
download_path = os.path.join(user, loc)
os.makedirs(download_path, exist_ok=True)

# Read date and Wayback item numbers
data_dict = {}
with open(file_path, "r") as file:
    for line in file:
        date, number = line.strip().split()
        data_dict[date] = int(number)

# Function to capture and save screenshots
def capture_screenshot(browser, date, num):
    url = f'https://livingatlas.arcgis.com/wayback/#active={num}&mapCenter={mapCord}'
    browser.get(url)
    time.sleep(2)
    
    try:
        browser.find_element(By.ID, "onetrust-accept-btn-handler").click()
        time.sleep(2)
    except Exception:
        pass
    
    try:
        browser.find_element(By.CSS_SELECTOR, "div.margin-left-half.margin-right-quarter.cursor-pointer").click()
        time.sleep(2)
    except Exception:
        pass
    
    screenshot_path = os.path.join(download_path, 'temp.png')
    browser.save_screenshot(screenshot_path)
    crop_and_save_image(screenshot_path, date)

# Function to crop and save images
def crop_and_save_image(image_path, date):
    left, upper, right, lower = 400, 60, 1910, 930
    img = Image.open(image_path).crop((left, upper, right, lower))
    img.save(os.path.join(download_path, f'{date}.png'))

# Main execution loop
for date, num in tqdm(data_dict.items(), desc="Capturing Screenshots"):
    browser = webdriver.Firefox()
    browser.maximize_window()
    capture_screenshot(browser, date, num)
    browser.quit()