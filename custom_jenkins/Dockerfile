# ==============================
# Jenkins + Docker-in-Docker
# ==============================

# Use the official Jenkins LTS image
FROM jenkins/jenkins:lts

# Switch to root to install dependencies
USER root

# Install dependencies required for Docker
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Docker dynamically (works on any Debian version)
RUN curl -fsSL https://get.docker.com | sh

# Add Jenkins user to the Docker group
RUN groupadd -f docker && usermod -aG docker jenkins

# Create Docker volume for DinD
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Optional: environment variable for Docker socket usage
ENV DOCKER_HOST=unix:///var/run/docker.sock

# Switch back to Jenkins user
USER jenkins
