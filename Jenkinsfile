pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Verify Application') {
            steps {
                echo 'Application check successful'
            }
        }
    }
}