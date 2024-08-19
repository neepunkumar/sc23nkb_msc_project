# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install the most common packages first
COPY requirements-base.txt requirements-base.txt
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements-base.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME TransformerAutoscaler

# Run transformer_service.py when the container launches
CMD ["python", "transformer_service.py"]
