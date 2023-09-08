# Since TensorFlow won't support CUDA 11 I've created this docker to
# run the models on my GPU instead of the CPU

# Use an official TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install Matplotlib using pip
RUN pip install matplotlib

# Set the working directory
WORKDIR /tf-gpu

# Copy your Python script (if you have one) into the container
COPY AlexNet.py /tf-gpu/
COPY LeNet.py /tf-gpu/
COPY VGG.py /tf-gpu/

# Command to run when the container starts
# CMD ["python", "AlexNet.py"]
# CMD ["python", "LeNet.py"]
# CMD ["python", "VGG.py"]
