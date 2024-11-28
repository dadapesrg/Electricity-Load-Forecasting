import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
url='https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub?embedded=true'
response = requests.get(url)

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Step 3: Find the table (you can use more specific selection criteria based on the table's id/class)
table = soup.find('table')
def convert_to_int_if_possible(value):
    try:
        # Attempt to convert the value to an integer
        return int(value)
    except ValueError:
        # If it fails, return the value as a string
        return value.strip()
cell_data = []
grid_data = []
for row in table.find_all('tr'):
    cols = row.find_all('td')
    print(cols)
    #grid_data.append((x, y, char))
   
    for col in cols:
        row_data = []
        row_data.append(col.text.strip())
  
    cell_data.append(row_data)
   # print(cell_data)
    #cell_data.append(cols[0], cols[1], cols[2])
 #   cell_data.append([col for col in cols])
cell_data = np.array(cell_data)
cell_data = pd.DataFrame(cell_data)
cell_data.to_csv('data/test.csv')
#grid_data.to_csv('data/test.csv')
# Step 5: Iterate through table rows and extract character and integer data
for row in table.find_all('tr'):
    cells = row.find_all('td')  # Extract all <td> elements (table cells)
  #  cell_data = []
    
    # Step 6: Process each cell's content
    for cell in cells:
        # Get the cell's text and try to convert it to int if possible
        data = cell.text.strip()
        converted_data = convert_to_int_if_possible(data)
        cell_data.append(converted_data)
cell_data = np.array(cell_data)
cell_data = pd.DataFrame(cell_data)
# cell_data.transpose()   
cell_data.to_csv('data/test.csv')

def print_secret_message(url):
    # Send a GET request to the Google Doc URL
    response = requests.get(url)
    
    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        page_content = response.content
        
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Find all the table rows in the document
       # table_rows = soup.find_all('tr')
        table = soup.find('table')
        for row in table.find_all('tr'):
            cells = row.find_all('td')  # Extract all <td> elements (table cells)
            for cell in cells:
                cell_data = cell.text.strip() #for cell in cells]  # Get text from each cell, and strip whitespace
                for data in cell_data:
                    print(data)
            # Get the text of the table data
                                                
                # Split the text into the character and its position
                    for x, y in enumerate(data):                
                        print(x, y) 
              #  print(cell_data)

        
        # Initialize an empty dictionary to store the characters and their positions
    grid_data = []
    for row in table_rows:
         # Find all the table data in the row
        table_data = row.find_all('td')
            
        # Iterate over each table data
        for data in table_data:
            # Get the text of the table data
            text = data.get_text()
                              
                # Split the text into the character and its position
            for x, char, y in enumerate(text):
                
                 print(x, y, char)                
                # Store the character and its position in the grid dictionary
                    
       #  print(grid)   
        # Find the maximum x and y coordinates
        #max_x = max(max(row.keys()) for row in grid.values())
        # max_y = max(grid.keys())
        #print (max_x, max_y)"

url = "https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub"
print_secret_message(url)

import gspread
from unicodedata import name

def read_google_doc(url):
    # Connect to the Google Doc
    gc = gspread.service_account(filename='') #filename='credentials.json')
    sh = gc.open_by_url(url)
    ws = sh.sheet1

    # Get the grid dimensions
    grid_size = ws.cell(1, 1).value.split('x')
    grid_width = int(grid_size[0])
    grid_height = int(grid_size[1])

    # Create the grid
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    # Fill in the characters
    for row in ws.get_all_values()[1:]:
        if len(row) == 4:
            char, x, y, _ = row
            x, y = int(x), int(y)
            if x < grid_width and y < grid_height:
                grid[y][x] = chr(int(name(char, '').split(' ')[0], 16))

    # Print the grid
    for row in grid:
        print(''.join(row))

# Example usage
read_google_doc('https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub')


import requests
from bs4 import BeautifulSoup

def fetch_google_doc_content(url):
    # Fetch the HTML content of the Google Doc page
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch document. Status code: {response.status_code}")

def parse_data(html_content):
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Assuming that the relevant data is in <p> tags or some structured format.
    # You will need to inspect the HTML structure of the Google Doc to find out where the data is located.
    # For this example, we'll assume the data is inside <p> tags in the format: "x y character"
    
    grid_data = []
    paragraphs = soup.find_all('p')
    
    for p in paragraphs:
        text = p.get_text().strip()
        if text:
            # Split the line into x, y, and character
            parts = text.split()
            x = int(parts[0])
            y = int(parts[1])
            char = parts[2]
            grid_data.append((x, y, char))
    
    return grid_data

def create_grid(grid_data):
    # Find the maximum x and y values to define the grid dimensions
    max_x = max([x for x, y, char in grid_data])
    max_y = max([y for x, y, char in grid_data])
    
    # Initialize the grid with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Place the characters in their respective positions
    for x, y, char in grid_data:
        grid[y][x] = char
    
    return grid

def print_grid(grid):
    # Print the grid row by row
    for row in grid:
        print(''.join(row))

def main(url):
    # Step 1: Fetch the Google Doc HTML content
    html_content = fetch_google_doc_content(url)
    
    # Step 2: Parse the content to get the grid data
    print(html_content)
    grid_data = parse_data(html_content)
    
    # Step 3: Create a grid and place the characters in their positions
    grid = create_grid(grid_data)
    
    # Step 4: Print the final grid
    print_grid(grid)

# Example usage
url = 'https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub'  # Replace with the actual Google Doc URL
main(url)



import requests

def fetch_google_doc_content(url):
    # Use requests to fetch the raw data from the URL (public Google Doc)
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch document. Status code: {response.status_code}")

def parse_data(doc_content):
    # Parse the content to extract characters and their coordinates.
    # Assuming the doc format is well-structured like:
    # x y character (one per line)
    # Example: 
    # 0 0 F
    # 1 0 F
    # ...
    
    grid_data = []
    lines = doc_content.splitlines()
    for line in lines:
        if line.strip():
            parts = line.split()
            x = int(parts[0])
            y = int(parts[1])
            char = parts[2]
            grid_data.append((x, y, char))
    
    return grid_data

def create_grid(grid_data):
    # Find the max x and y to define the grid dimensions
    max_x = max([x for x, y, char in grid_data])
    max_y = max([y for x, y, char in grid_data])
    
    # Initialize the grid with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Place the characters at their corresponding positions
    for x, y, char in grid_data:
        grid[y][x] = char
    
    return grid

def print_grid(grid):
    # Print the 2D grid in a fixed-width format
    for row in grid:
        print(''.join(row))

def main(url):
    # Step 1: Fetch document content
    doc_content = fetch_google_doc_content(url)
    
    # Step 2: Parse the content to get character positions
    grid_data = parse_data(doc_content)
    
    # Step 3: Create a 2D grid and place the characters
    grid = create_grid(grid_data)
    
    # Step 4: Print the final grid
    print_grid(grid)

# Example usage with a URL (assuming public access to the document)
url = 'https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub'
main(url)

import gspread
from unicodedata import name

def read_google_doc(url):
    # Connect to the Google Doc
    gc = gspread.service_account() #filename='credentials.json')
    sh = gc.open_by_url(url)
    ws = sh.sheet1

    # Get the grid dimensions
    grid_size = ws.cell(1, 1).value.split('x')
    grid_width = int(grid_size[0])
    grid_height = int(grid_size[1])

    # Create the grid
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    # Fill in the characters
    for row in ws.get_all_values()[1:]:
        if len(row) == 4:
            char, x, y, _ = row
            x, y = int(x), int(y)
            if x < grid_width and y < grid_height:
                grid[y][x] = chr(int(name(char, '').split(' ')[0], 16))

    # Print the grid
    for row in grid:
        print(''.join(row))

# Example usage
read_google_doc('https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub')

#main(url)
