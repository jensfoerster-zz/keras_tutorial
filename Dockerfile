# Use an official Python runtime and tensorflow parent images
FROM python:2.7-slim
FROM tensorflow/tensorflow:latest

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
#VOLUME [ "/app" ]
ADD . /app

# Install dependencies
RUN apt-get update && apt-get install -y \
  python-tk

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
#EXPOSE 80

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]