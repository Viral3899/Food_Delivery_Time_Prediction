END APPLICATION SANPSHOT

![image](https://user-images.githubusercontent.com/101617198/236999609-882fad2f-6498-4cbe-8084-4df09761184b.png)

![image](https://user-images.githubusercontent.com/101617198/236999651-5301c1bc-a0d8-4bbe-b819-4a344a88d9e1.png)


**RESULTS**

![image](https://user-images.githubusercontent.com/101617198/236999687-64f26e29-3c5c-4e94-b769-1a728a3f81db.png)

RETRAIN 

![image](https://user-images.githubusercontent.com/101617198/236999664-b29c9e1f-5529-4a15-88ca-e0be23e3db8a.png)

# GENERAL STRUCTURE 

![Slide1](https://user-images.githubusercontent.com/101617198/230705112-297b86b4-7fb0-43c5-b40e-b849c0dd064e.JPG)




## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade -y 
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	   
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-2

    AWS_ECR_LOGIN_URI = demo>> 015889189520.dkr.ecr.us-east-2.amazonaws.com

    ECR_REPOSITORY_NAME = foodapprepo


