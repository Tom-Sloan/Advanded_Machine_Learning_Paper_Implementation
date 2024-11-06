import markdown
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests

# Specify the path to your markdown file
file_path = '_ Github.md'

# Initialize an empty array
lines_with_round_brackets = []

# Read the markdown file
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        
        # Iterate over each line in the file
        for line in file:
            # Check if the line contains round brackets
            if '(' in line or ')' in line:
                # Add the line to the array
                # Extract the text between the round brackets
                text_between_round_brackets = line[line.index('(') + 1:line.index(')')]

                # Add the extracted text to the array
                lines_with_round_brackets.append(text_between_round_brackets)

# Print out the array with index
for i, item in enumerate(lines_with_round_brackets):
    print(f"Index {i}: {item}")
def sanitize_filename(filename):
    # Replace colons with underscores
    sanitized = filename.replace(":", "_")
    # Truncate to 255 characters if necessary
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized

# Initialize Safari WebDriver
driver = webdriver.Safari()

# Iterate over each link in the array
for link in lines_with_round_brackets:
    # Check if the link is from arxiv
    if 'arxiv.org' in link:
        continue
        try:
            try:
                # Open the link in the webdriver
                driver.get(link)
                
                # Wait for title to appear
                title_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.title')))
                
                # Extract the title of the page
                title = title_element.text
                title = sanitize_filename(title)

                print(f"Downloading {title}")
            except:
                print(f"Error getting title {link}")
            
            link = link.replace('abs', 'pdf')
            # Extract the filename from the URL
            filename = title + '.pdf'

            # Specify the path to the subfolder
            subfolder_path = '/Users/Sloan/Desktop/Project_Desktop/Rafik/lidar_world/PDFs/Downloaded_Pages/'

            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Specify the path to save the downloaded page
            save_path = os.path.join(subfolder_path, filename)

            # Download the PDF file directly using the requests library
            response = requests.get(link)

            # Save the PDF to a file
            with open(save_path, "wb") as file:
                file.write(response.content)
        except:
            print(f"Error downloading {link}")
    elif 'openaccess' in link:
        try:
            try:
                # Open the link in the webdriver
                driver.get(link)
                
                # Wait for title to appear
                title_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#papertitle')))
                
                # Extract the title of the page
                title = title_element.text
                title = sanitize_filename(title)
                title = title.strip()

                print(f"Downloadin{title}asd")
            except:
                print(f"Error getting title {link}")
            
            link = link.replace('/html/', '/papers/')
            link = link.replace('.html', '.pdf')
            # Extract the filename from the URL
            filename = title + '.pdf'

            # Specify the path to the subfolder
            subfolder_path = '/Users/Sloan/Desktop/Project_Desktop/Rafik/lidar_world/PDFs/Downloaded_Pages/'

            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Specify the path to save the downloaded page
            save_path = os.path.join(subfolder_path, filename)

            # Download the PDF file directly using the requests library
            response = requests.get(link)

            # Save the PDF to a file
            with open(save_path, "wb") as file:
                file.write(response.content)
        except:
            print(f"Error downloading {link}")
    elif 'ecva.net' in link and 'pdf' in link:
        try:
            subfolder_path = '/Users/Sloan/Desktop/Project_Desktop/Rafik/lidar_world/PDFs/Downloaded_Pages/'
            filename = link.split('/')[-1]
            save_path = os.path.join(subfolder_path, filename)
            response = requests.get(link)
            with open(save_path, "wb") as file:
                file.write(response.content)
        except:
            print(f"Error downloading {link}")
# Close the webdriver
driver.quit()




