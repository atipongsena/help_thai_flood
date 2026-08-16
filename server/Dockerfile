# Use Node.js 18 Alpine image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Install typescript globally to avoid permission issues
RUN npm install -g typescript

# Copy source code
COPY . .

# Build the application
RUN tsc

# Expose port 4000
EXPOSE 4000

# Run the application
CMD ["npm", "start"]
