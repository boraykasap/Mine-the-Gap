# My Static Website

This repository contains a simple HTML + JavaScript page that can be hosted in a Docker container using Nginx.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your system.

## Build and Run the Docker Container

1. **Build the Docker image**  

a) Open a terminal in the project directory and run:

```sh
   docker build -t my-static-site .
```
b) Then this

```sh
docker run -d -p 8080:80 my-static-site
```
c) open: http://localhost:8080