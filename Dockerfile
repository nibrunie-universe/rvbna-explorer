FROM python:3.14-slim

# Install system dependencies needed for pysollya
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libmpfr-dev \
    libmpfi-dev \
    libfplll-dev \
    libxml2-dev \
    zlib1g-dev \
    autoconf \
    automake \
    libtool \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
