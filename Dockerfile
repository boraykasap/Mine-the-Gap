# Use Node.js official image
FROM node:18-alpine

# Set working directory inside container
WORKDIR /app

# Copy package.json and package-lock.json first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the source code
COPY . .

# Expose the port your app runs on (usually 3000 or 5173 depending on framework)
EXPOSE 8080

# Run the app
CMD ["npm", "run", "dev"]