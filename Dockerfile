# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /research

# Install system dependencies required by the 'unstructured' library
# This is crucial for handling file types like PDFs correctly.
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container first
# This leverages Docker's layer caching.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
# This includes app.py and the .streamlit directory.
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Set Streamlit-specific environment variables
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_HEADLESS true

# The command to run your Streamlit app when the container launches
CMD ["streamlit", "run", "main.py"]
