import os
import requests
from bs4 import BeautifulSoup

# URL of the website containing the .zip files
URL = "https://www.bts.gov/browse-statistical-products-and-data/bts-publications/data-bank-28dm-t-100-domestic-market-data"

# Directory to save the downloaded .zip files
DOWNLOAD_DIR = "c:\\Users\\Steven\\Desktop\\school github repos\\svo-directed-practicum\\Data\\Downloaded_Zips"

# Create the download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_zip_files():
    try:
        # Fetch the webpage content
        response = requests.get(URL)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links ending with .zip
        zip_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.zip')]

        if not zip_links:
            print("No .zip files found on the webpage.")
            return

        # Download each .zip file
        for link in zip_links:
            file_name = os.path.basename(link)
            file_path = os.path.join(DOWNLOAD_DIR, file_name)

            print(f"Downloading {file_name}...")
            zip_response = requests.get(link)
            zip_response.raise_for_status()

            # Save the .zip file
            with open(file_path, 'wb') as file:
                file.write(zip_response.content)

            print(f"Saved {file_name} to {DOWNLOAD_DIR}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_zip_files()
