pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins...............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Monoj10das/MLOPS-COURSE-PROJECT-1.git']])
                }
            }
        }

        stage('Setting up our Virtual Environment and installing dependencies'){
            steps{
                script{
                    echo 'Setting up our Virtual Environment and installing dependencies'
                    sh '''
                        python -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -e .
                   '''
                }
            }
        }
    }
}
