FROM centos:7

RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
    java-1.8.0-openjdk wget

RUN python -V 
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip 
RUN pip3 install numpy pandas pyspark

RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
    && mkdir -p /opt/spark \
    && tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
    && rm apache-spark.tgz

RUN ln -s /opt/spark-3.1.2-bin-hadoop2.7 /opt/spark
RUN echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc \
    && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc \
    && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc \
    && source ~/.bashrc

RUN mkdir -p /wpr/src /wpr/src/test_model

COPY src/winetestdata_prediction.py /wpr/src/
COPY src/test_model/ /wpr/src/test_model/

WORKDIR /wpr/src/

ENTRYPOINT ["python3", "winetestdata_prediction.py"]