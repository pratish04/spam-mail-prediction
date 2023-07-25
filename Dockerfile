# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the app files (assuming your Flask app code is in a directory named "flask_app" in your project)
COPY flask_app /app

# Install dependencies (if you have a requirements.txt file)
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Set the Flask app entry point (assuming your main Flask app file is named "app.py")
CMD ["python", "app.py"]
