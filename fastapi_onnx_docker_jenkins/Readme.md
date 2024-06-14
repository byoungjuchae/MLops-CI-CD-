This folder use jenkins and docker.


First, We execute the jenkins.

```
docker pull jenkins/jenkins:lts-jdk17
```
Start the jenkins
```
docker run -itd --name jenkins -p 8080:8080 -p 50000:50000 -v /home/dexter/:/var/run/docker.sock -e TZ=Asia/Seoul -u root jenkins/jenkins:lts-jdk17
```
And install the jenkins.

Second, We make a docker image in the app folder.

```
docker build -t practice .

```

if we use the kubernetes or docker compose, it will be more convenient.

