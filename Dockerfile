# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY . .
# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container


# Expose the port that the Flask app will run on
EXPOSE 8080

# Set the environment variables
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
