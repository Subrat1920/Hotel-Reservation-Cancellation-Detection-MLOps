pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "hotel-reservation-1920"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
        REGION = "us-central1"
        SERVICE_NAME = "hotel-reservation-prediction"
    }

    stages {

        stage('Clone GitHub Repo') {
            steps {
                script {
                    echo 'Cloning GitHub repo...'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/Subrat1920/Hotel-Reservation-Cancellation-Detection-MLOps.git'
                        ]]
                    )
                }
            }
        }

        stage('Create Virtual Environment & Install Dependencies') {
            steps {
                script {
                    echo 'Setting up virtual environment and installing dependencies...'
                    sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate

                    pip install --upgrade pip
                    pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Build & Push Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Building and pushing Docker image to GCR...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:latest .
                        docker push gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:latest
                        '''
                    }
                }
            }
        }

        stage('Deploy to Google Cloud Run') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Deploying to Google Cloud Run...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        # Delete existing service if it exists to avoid conflicts
                        if gcloud run services describe ${SERVICE_NAME} --platform=managed --region=${REGION} > /dev/null 2>&1; then
                            echo "Service ${SERVICE_NAME} exists. Deleting..."
                            gcloud run services delete ${SERVICE_NAME} --platform=managed --region=${REGION} --quiet
                        fi

                        # Deploy new service
                        gcloud run deploy ${SERVICE_NAME} \
                            --image=gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:latest \
                            --platform=managed \
                            --region=${REGION} \
                            --allow-unauthenticated \
                            --quiet
                        '''
                    }
                }
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline executed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for details.'
        }
    }
}
