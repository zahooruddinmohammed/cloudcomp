# AWS Spark Wine Quality Prediction Application

Cloud-Comp is a PySpark application designed to parallelly train a machine learning model on EC2 instances for predicting wine quality using publicly available data. The trained model is then utilized to predict wine quality.

This project incorporates Docker to create a container for the trained machine learning model, streamlining deployment processes.

The primary Python source files in this project are:

1. `wine_prediction.py`: Reads the training dataset from S3 and trains the model in parallel on an EMR Spark cluster. Once trained, the model can be executed on provided test data via S3. The program stores the trained model in the S3 bucket (Public URL for the bucket - S3://wine-data-12).


2. `wine_test_data_prediction.py`: Loads the trained model and executes it on a given test data file. This program prints the F1 score as a metric for the accuracy of the trained model.

Dockerfile: Used to create a Docker image and run a container for simplified deployment.


## Instruction to use:

### 1. How to create a Spark cluster in AWS

Users can create a Spark cluster using the EMR console provided by AWS. Follow these steps to create one with 4 EC2 instances (adjust the number of instances based on your load).

1. Create a Key-Pair for the EMR cluster via navigation `EC2 -> Network & Security -> Key-pairs`. Use `.pem` as the format, which will download the `<name of key pair>.pem` file. Keep it safe as it will be needed for SSH to EC2 instances.

2. Navigate to the Amazon EMR console using the link [EMR Console](https://console.aws.amazon.com/elasticmapreduce/home?region=us-east-1). Then, navigate to `Clusters -> Create Cluster`.

3. Fill in the respective sections:
   ```
   General Configuratin -> Cluster Name 
   Software Configuration-> EMR 5.33 , do select 'Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.9.0' option menu.
   Harware Configuration -> Make instance count as 4
   Security Access -> Provide .pem key created in above step.
   Rest of parameters can be left default.
   ```


   
A cluster can alos be created using cli command:
   ```
  aws emr create-cluster --applications Name=Spark Name=Zeppelin --ebs-root-volume-size 10 --ec2-attributes '{"KeyName":"ec2-spark","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-42c0ca0f","EmrManagedSlaveSecurityGroup":"sg-0d7ed2552ba71f5af","EmrManagedMasterSecurityGroup":"sg-0e853f0a4bdc5f799"}' --service-role EMR_DefaultRole --enable-debugging --release-label emr-5.33.0 --log-uri 's3n://aws-logs-367626191020-us-east-1/elasticmapreduce/' --name 'My cluster' --instance-groups '[{"InstanceCount":3,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core Instance Group"},{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master Instance Group"}]' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1
  ```
  

4. Cluster status should be 'Waiting ' on successful cluster creation.
 ![image](https://github.com/zahooruddinmohammed/Cloud-comp/assets/130806627/7f692eb1-9054-43ac-9d28-1444b612f1ee)
## Training ML model in spark cluster with 4ec2 instances in parallel.
  1.  when cluster is ready to accept jobs, submit one you can either use step button to add steps or submit manually. perfroming SSH to Master of cluster using below command:
     ```
     ssh -i "ec2key.pem" <<User>>@<<Public IPv4 DNS>> ```
![image](https://github.com/zahooruddinmohammed/Cloud-comp/assets/130806627/29200515-c8c0-4bc8-a837-a0b18f3de3fe)


  3. Upon successful login to the master, switch to the root user by running the command:
 ```
  sudo su
  ```
![image](https://github.com/zahooruddinmohammed/Cloud-comp/assets/130806627/334ec44c-41e9-4ac5-b410-cd3bf8b23274)

   3. Submit the job using the following command:
  ```
   spark-submit s3://zm254/wine_prediction.py
 ```
![image](https://github.com/zahooruddinmohammed/Cloud-comp/assets/130806627/8c79b532-7705-4aa5-b439-4eed1b8621f7)
![image](https://github.com/zahooruddinmohammed/Cloud-comp/assets/130806627/2e056d28-aea5-43b6-a123-1af9d6002551)



4. You can trace the status of this job in the EMR UI application logs. Once the status is successful, a `test.model` will be created in the S3 bucket - `s3://wine-data-12`.

### 3. How to run the trained ML model locally without Docker

1. Clone this repository.
2. Ensure that you have a Spark environment set up locally. To set up one, follow the link [Spark Setup](https://spark.apache.org/docs/latest).
3. Navigate to `cloud-comp/src` folder.
4. Place your trainingdataset in the 'cloud-comp/data/csv' folder.
5. Install PySpark; you can use `pip install pyspark` or install it via `conda`.
6. Once the setup is ready, execute the following command:
  
 ``` 
 cd cloud-comp/src
 spark-submit wine_test_data_prediction.py <filename>
 ```
 ### 4. Run ML model using Docker

1. Install Docker where you want to run this container.
2. A public image has been created and posted on DockerHub. Use the command `docker pull zahooruddin/cloud_comp` to get the image on your machine.
3. Place your test data file in a folder (let's call it directory `dirA`), which you will mount with the Docker container.
4. Run the Docker image:docker run -v {directory path for data dirA}:/code/data/csv zahooruddin/cloud_comp {testdata file name}

Sample command
```
docker run -v /Users/<username>/<path-to-folder>/cloud_comp/data/csv:/code/data/csv zahooruddin/cloud_comp trainingdataset.csv



```

![image](https://github.com/zahooruddinmohammed/cloudcomp/assets/130806627/fd4e7321-3350-41b4-8ead-a8c064462fcc)
### Docker URL
https://hub.docker.com/layers/zahooruddin029/wpr/v1/images/sha256-ddb09a6bf19f9d8444f4529c3b9b5f518af2575e8a392e7778b79eace66731de?context=repo
