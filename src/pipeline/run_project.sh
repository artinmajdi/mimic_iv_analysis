#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MIMIC-IV Data Pipeline Project Setup ===${NC}"
echo -e "${BLUE}This script will help you run the project using either Docker or a virtual environment.${NC}"
echo

# Function to create .env file if it doesn't exist
create_env_file() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        cat > .env << EOL
# Environment variables for MIMIC-IV Data Pipeline
# Add your configuration variables below

# Example:
# DATA_PATH=/path/to/data
# MODEL_OUTPUT_PATH=/path/to/output
EOL
        echo -e "${GREEN}.env file created successfully.${NC}"
    else
        echo -e "${GREEN}.env file already exists.${NC}"
    fi
}

# Function to setup virtual environment
setup_venv() {
    echo -e "${BLUE}Setting up virtual environment...${NC}"

    # Check if python or python3 is installed
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Neither python nor python3 is installed. Please install Python and try again.${NC}"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo -e "${RED}Error: Python 3.12 or higher is required. You have Python $PYTHON_VERSION.${NC}"
        echo -e "${YELLOW}Please install a newer version of Python and try again.${NC}"
        exit 1
    fi

    # Create .env file if it doesn't exist
    create_env_file

    # Check if .venv directory exists
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        $PYTHON_CMD -m venv .venv
    fi

    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}uv is not installed. Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install uv. Falling back to pip.${NC}"
            # Check if pip or pip3 is available
            if command -v pip &> /dev/null; then
                PIP_CMD="pip"
            elif command -v pip3 &> /dev/null; then
                PIP_CMD="pip3"
            else
                echo -e "${RED}Error: Neither pip nor pip3 is available. Cannot continue.${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}uv installed successfully.${NC}"
            PIP_CMD="uv pip"
        fi
    else
        echo -e "${GREEN}uv is already installed.${NC}"
        PIP_CMD="uv pip"
    fi

    # Upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    $PIP_CMD install --upgrade pip > /dev/null 2>&1
    echo -e "${GREEN}Pip upgraded successfully.${NC}"

    # Install requirements
    echo -e "${YELLOW}Installing requirements...${NC}"
    echo -e "${YELLOW}This may take a few minutes. Please be patient...${NC}"
    
    # Create a simple progress indicator
    echo -n "${YELLOW}Installing packages: ${NC}"
    $PIP_CMD install -r config/requirements.txt > /tmp/pip_install.log 2>&1 &
    PIP_PID=$!
    
    # Show a spinner while pip is running
    spin='-\|/'
    i=0
    while kill -0 $PIP_PID 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        printf "\r${YELLOW}Installing packages: ${spin:$i:1}${NC}"
        sleep .3
    done
    
    # Check if pip install was successful
    wait $PIP_PID
    if [ $? -eq 0 ]; then
        printf "\r${GREEN}Requirements installed successfully!     ${NC}\n"
    else
        printf "\r${RED}Error installing requirements. Check the log at /tmp/pip_install.log${NC}\n"
        exit 1
    fi

    echo -e "${GREEN}Virtual environment setup complete!${NC}"
    echo -e "${YELLOW}To activate the virtual environment in the future, run:${NC}"
    echo -e "source .venv/bin/activate"
    echo
    echo -e "${YELLOW}To deactivate the virtual environment, run:${NC}"
    echo -e "deactivate"
}

# Function to setup and run Docker
setup_docker() {
    echo -e "${BLUE}Setting up Docker...${NC}"

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed. Please install it and try again.${NC}"
        echo -e "${YELLOW}You can install Docker from https://docs.docker.com/get-docker/${NC}"
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running. Please start Docker and try again.${NC}"
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}Checking for Docker Compose plugin...${NC}"
        if ! docker compose version &> /dev/null; then
            echo -e "${RED}Error: Neither docker-compose nor docker compose plugin is installed.${NC}"
            echo -e "${YELLOW}You can install Docker Compose from https://docs.docker.com/compose/install/${NC}"
            exit 1
        else
            echo -e "${GREEN}Docker Compose plugin is available.${NC}"
            COMPOSE_CMD="docker compose"
        fi
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Create .env file if it doesn't exist
    create_env_file

    # Check if container is already running
    CONTAINER_NAME="mimic-iv-pipeline"
    if docker ps | grep -q $CONTAINER_NAME; then
        echo -e "${YELLOW}Container is already running. Do you want to:${NC}"
        echo -e "1) Connect to the running container"
        echo -e "2) Stop the container and start a new one"
        echo -e "3) Exit"
        read -p "Enter your choice (1-3): " container_choice

        case $container_choice in
            1)
                echo -e "${YELLOW}Connecting to running container...${NC}"
                echo -e "${GREEN}Jupyter notebook should be available at http://localhost:8888${NC}"
                echo -e "${YELLOW}Press Ctrl+C to exit this view (container will keep running)${NC}"
                docker logs -f $CONTAINER_NAME
                return
                ;;
            2)
                echo -e "${YELLOW}Stopping container...${NC}"
                docker stop $CONTAINER_NAME
                ;;
            3)
                echo -e "${BLUE}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting...${NC}"
                exit 1
                ;;
        esac
    fi

    # Check if Docker image exists
    IMAGE_NAME="mimic-iv-data-pipeline"
    if docker images | grep -q $IMAGE_NAME; then
        echo -e "${GREEN}Docker image already exists.${NC}"
    else
        echo -e "${YELLOW}Building Docker image...${NC}"
        echo -e "${YELLOW}This may take a few minutes. Please be patient...${NC}"
        
        # Run docker-compose build with progress indicator
        echo -n "${YELLOW}Building image: ${NC}"
        $COMPOSE_CMD -f config/docker-compose.yml build > /tmp/docker_build.log 2>&1 &
        BUILD_PID=$!
        
        # Show a spinner while building
        spin='-\|/'
        i=0
        while kill -0 $BUILD_PID 2>/dev/null; do
            i=$(( (i+1) % 4 ))
            printf "\r${YELLOW}Building image: ${spin:$i:1}${NC}"
            sleep .3
        done
        
        # Check if build was successful
        wait $BUILD_PID
        if [ $? -eq 0 ]; then
            printf "\r${GREEN}Docker image built successfully!     ${NC}\n"
        else
            printf "\r${RED}Error building Docker image. Check the log at /tmp/docker_build.log${NC}\n"
            exit 1
        fi
    fi

    # Run Docker container
    echo -e "${YELLOW}Starting Docker container...${NC}"
    echo -e "${GREEN}Jupyter notebook will be available at http://localhost:8888${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the container.${NC}"
    
    # Ask if user wants to run in detached mode
    echo -e "${YELLOW}Do you want to run the container in the background? (y/n)${NC}"
    read -p "Enter your choice (y/n): " detached_choice
    
    case $detached_choice in
        [Yy])
            $COMPOSE_CMD -f config/docker-compose.yml up -d
            echo -e "${GREEN}Container started in background.${NC}"
            echo -e "${YELLOW}To view logs, run: ${NC}docker logs -f mimic-iv-pipeline"
            echo -e "${YELLOW}To stop the container, run: ${NC}$COMPOSE_CMD -f config/docker-compose.yml down"
            ;;
        *)
            $COMPOSE_CMD -f config/docker-compose.yml up
            ;;
    esac
}

# Main menu
echo -e "${YELLOW}Please select an option:${NC}"
echo -e "1) Setup and run with virtual environment"
echo -e "2) Setup and run with Docker"
echo -e "3) Exit"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        setup_venv
        ;;
    2)
        setup_docker
        ;;
    3)
        echo -e "${BLUE}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting...${NC}"
        exit 1
        ;;
esac
