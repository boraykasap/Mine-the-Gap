# Use the official Nginx image as the base
FROM nginx:alpine

# Remove default Nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy all files into Nginx's default directory
COPY . /usr/share/nginx/html/

# Expose port 80
EXPOSE 8080

# Start Nginx when the container starts
CMD ["sh", "-c", "nginx -g 'daemon off;' & echo 'Your site is running at http://localhost:8080' && wait"]