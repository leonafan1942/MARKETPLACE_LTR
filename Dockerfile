# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables for model saving
ENV MODEL_DIR=/app/models
ENV MODEL_FILE=listnet_wb.pth

# Create the model directory
RUN mkdir -p $MODEL_DIR

# Expose any ports the app runs on
EXPOSE 5000

# Define default command to run on container start
CMD ["python", "train.py"]
