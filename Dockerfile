# Use an official Python runtime as the base image
FROM python:3.7


COPY . /app
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
RUN pip install -r requirements.txt

# Copy the Flask app code into the container
# Expose the port on which your Flask app runs
EXPOSE 5000
# Start the Flask application
CMD ["python", "app.py"]
