import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('instance/users.db')  # Replace with the actual path to your database

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Fetch all rows from the 'user' table
cursor.execute("SELECT * FROM user")

# Fetch all rows of the table
rows = cursor.fetchall()

# Get column names from the cursor description
column_names = [description[0] for description in cursor.description]

# Display the column names
print("Columns:", column_names)

# Display all rows with their column content
print("\nData in 'user' table:")
for row in rows:
    # Each 'row' is a tuple, where each element corresponds to a column's data in that row
    print(row)

# Close the connection
conn.close()

# import sqlite3

# # Connect to the SQLite database
# conn = sqlite3.connect('instance/users.db')

# # Create a cursor object to execute SQL queries
# cursor = conn.cursor()

# # Define the new values and the ID of the user you want to update
# new_name = "Marvel"
# new_email = "patildhiraj2357@gmail.com"
# user_id = 1  # Change this to the ID of the user you want to update

# # Update the user's name and email
# cursor.execute("""
#     UPDATE user
#     SET name = ?, email = ?
#     WHERE id = ?
# """, (new_name, new_email, user_id))

# # Commit the changes
# conn.commit()

# print(f"User with ID {user_id} updated successfully.")

# # Close the connection
# conn.close()



# import os
# import fitz  # PyMuPDF for PDF extraction
# from docx import Document  # For DOCX file extraction

# def extract_text(file_path):
#     """Extract text from pdf, docx, or txt files."""
#     ext = file_path.split('.')[-1].lower()
#     text = ""

#     # Extract text from PDF
#     if ext == "pdf":
#         try:
#             # Open the PDF file
#             doc = fitz.open(file_path)
#             for page_num in range(doc.page_count):  # Loop through all pages
#                 page = doc.load_page(page_num)
#                 text += page.get_text("text")  # Extract raw text from each page
#         except Exception as e:
#             print(f"Error processing PDF file {file_path}: {e}")

#     # Extract text from DOCX
#     elif ext == "docx":
#         try:
#             # Open the DOCX file
#             doc = Document(file_path)
#             for para in doc.paragraphs:
#                 text += para.text + "\n"  # Append each paragraph's text
#         except Exception as e:
#             print(f"Error processing DOCX file {file_path}: {e}")

#     # Extract text from TXT
#     elif ext == "txt":
#         try:
#             # Open the TXT file and read its content
#             with open(file_path, "r", encoding="utf-8") as f:
#                 text = f.read()
#         except Exception as e:
#             print(f"Error processing TXT file {file_path}: {e}")

#     else:
#         print(f"Unsupported file format: {file_path}")
#         return None

#     return text.strip()  # Remove leading/trailing whitespace

# def process_files(input_folder):
#     """Process files from a folder and extract text."""
#     results = []

#     supported_extensions = ['pdf', 'docx', 'txt']  # Define supported file extensions

#     for root, dirs, files in os.walk(input_folder):
#         for filename in files:
#             file_ext = filename.split('.')[-1].lower()  # Get file extension

#             if file_ext not in supported_extensions:
#                 continue  # Skip unsupported file types

#             file_path = os.path.join(root, filename)
#             try:
#                 print(f"Processing file: {filename}")
#                 # Extract text from the file
#                 text = extract_text(file_path)
#                 if not text:
#                     continue

#                 # Append the results
#                 results.append({
#                     "filename": filename,
#                     "extracted_text": text
#                 })

#                 # Optionally save the extracted text to a new file
#                 with open(os.path.join(root, f"{filename}_extracted.txt"), "w", encoding="utf-8") as f:
#                     f.write(text)

#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")

#     return results

# # Usage: Provide the input folder containing the PDF, DOCX, and TXT files
# input_folder = "model-test/"  # Replace with your input folder
# results = process_files(input_folder)

# # Optionally, print out the results
# for result in results:
#     print(f"Filename: {result['filename']}")
#     print(f"/nExtracted Text: {result['extracted_text']}...")  # Print a snippet of the text (first 200 characters)
