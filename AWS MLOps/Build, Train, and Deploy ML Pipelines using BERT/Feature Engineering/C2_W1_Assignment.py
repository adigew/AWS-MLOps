#!/usr/bin/env python
# coding: utf-8

# # Feature transformation with Amazon SageMaker processing job and Feature Store
# 
# ### Introduction
# 
# In this lab you will start with the raw [Women's Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) dataset and prepare it to train a BERT-based natural language processing (NLP) model. The model will be used to classify customer reviews into positive (1), neutral (0) and negative (-1) sentiment.
# 
# You will convert the original review text into machine-readable features used by BERT. To perform the required feature transformation you will configure an Amazon SageMaker processing job, which will be running a custom Python script.
# 
# ### Table of Contents
# 
# - [1. Configure the SageMaker Feature Store](#c2w1-1.)
#   - [1.1. Configure dataset](#c2w1-1.1.)
#   - [1.2. Configure the SageMaker feature store](#c2w1-1.2.)
#     - [Exercise 1](#c2w1-ex-1)
# - [2. Transform the dataset](#c2w1-2.)
#     - [Exercise 2](#c2w1-ex-2)
#     - [Exercise 3](#c2w1-ex-3)
# - [3. Query the Feature Store](#c2w1-3.)
#   - [3.1. Export training, validation, and test datasets from the Feature Store](#c2w1-3.1.)
#     - [Exercise 4](#c2w1-ex-4)
#   - [3.2. Export TSV from Feature Store](#c2w1-3.2.)
#   - [3.3. Check that the dataset in the Feature Store is balanced by sentiment](#c2w1-3.3.)
#     - [Exercise 5](#c2w1-ex-5)
#     - [Exercise 6](#c2w1-ex-6)
#     - [Exercise 7](#c2w1-ex-7)
# 
# 

# In[2]:


# please ignore warning messages during the installation
get_ipython().system('pip install --disable-pip-version-check -q sagemaker==2.35.0')
get_ipython().system('conda install -q -y pytorch==1.6.0 -c pytorch')
get_ipython().system('pip install --disable-pip-version-check -q transformers==3.5.1')


# In[3]:


import boto3
import sagemaker
import botocore

config = botocore.config.Config(user_agent_extra='dlai-pds/c2/w1')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker', 
                  config=config)

featurestore_runtime = boto3.client(service_name='sagemaker-featurestore-runtime', 
                                    config=config)

sess = sagemaker.Session(sagemaker_client=sm,
                         sagemaker_featurestore_runtime_client=featurestore_runtime)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name


# <a name='c2w1-1.'></a>
# # 1. Configure the SageMaker Feature Store

# <a name='c2w1-1.1.'></a>
# ### 1.1. Configure dataset
# The raw dataset is in the public S3 bucket. Let's start by specifying the S3 location of it:

# In[4]:


raw_input_data_s3_uri = 's3://dlai-practical-data-science/data/raw/'
print(raw_input_data_s3_uri)


# List the files in the S3 bucket (in this case it will be just one file):

# In[5]:


get_ipython().system('aws s3 ls $raw_input_data_s3_uri')


# <a name='c2w1-1.2.'></a>
# ### 1.2. Configure the SageMaker feature store
# 
# As the result of the transformation, in addition to generating files in S3 bucket, you will also save the transformed data in the **Amazon SageMaker Feature Store** to be used by others in your organization, for example. 
# 
# To configure a Feature Store you need to setup a **Feature Group**. This is the main resource containing all of the metadata related to the data stored in the Feature Store. A Feature Group should contain a list of **Feature Definitions**. A Feature Definition consists of a name and the data type. The Feature Group also contains an online store configuration and an offline store configuration controlling where the data is stored. Enabling the online store allows quick access to the latest value for a record via the [GetRecord API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_feature_store_GetRecord.html). The offline store allows storage of the data in your S3 bucket. You will be using the offline store in this lab.
# 
# Let's setup the Feature Group name and the Feature Store offline prefix in S3 bucket (you will use those later in the lab):

# In[6]:


import time
timestamp = int(time.time())

feature_group_name = 'reviews-feature-group-' + str(timestamp)
feature_store_offline_prefix = 'reviews-feature-store-' + str(timestamp)

print('Feature group name: {}'.format(feature_group_name))
print('Feature store offline prefix in S3: {}'.format(feature_store_offline_prefix))


# Taking two features from the original raw dataset (`Review Text` and `Rating`), you will transform it preparing to be used for the model training and then to be saved in the Feature Store. Here you will define the related features to be stored as a list of `FeatureDefinition`.

# In[7]:


from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

feature_definitions= [
    # unique ID of the review
    FeatureDefinition(feature_name='review_id', feature_type=FeatureTypeEnum.STRING), 
    # ingestion timestamp
    FeatureDefinition(feature_name='date', feature_type=FeatureTypeEnum.STRING),
    # sentiment: -1 (negative), 0 (neutral) or 1 (positive). It will be found the Rating values (1, 2, 3, 4, 5)
    FeatureDefinition(feature_name='sentiment', feature_type=FeatureTypeEnum.STRING), 
    # label ID of the target class (sentiment)
    FeatureDefinition(feature_name='label_id', feature_type=FeatureTypeEnum.STRING),
    # reviews encoded with the BERT tokenizer
    FeatureDefinition(feature_name='input_ids', feature_type=FeatureTypeEnum.STRING),
    # original Review Text
    FeatureDefinition(feature_name='review_body', feature_type=FeatureTypeEnum.STRING),
    # train/validation/test label
    FeatureDefinition(feature_name='split_type', feature_type=FeatureTypeEnum.STRING)
]


# <a name='c2w1-ex-1'></a>
# 
# Create the feature group using the feature definitions defined above.
# 
# **Instructions:** Use the `FeatureGroup` function passing the defined above feature group name and the feature definitions.
# 
# ```python
# feature_group = FeatureGroup(
#     name=..., # Feature Group name
#     feature_definitions=..., # a list of Feature Definitions
#     sagemaker_session=sess # SageMaker session
# )
# ```

# In[8]:


from sagemaker.feature_store.feature_group import FeatureGroup

feature_group = FeatureGroup(
    name=feature_group_name,
    feature_definitions=feature_definitions,
    sagemaker_session=sess
)

print(feature_group)


# You will use the defined Feature Group later in this lab, the actual creation of the Feature Group will take place in the processing job. Now let's move into the setup of the processing job to transform the dataset.

# <a name='c2w1-2.'></a>
# # 2. Transform the dataset
# 
# You will configure a SageMaker processing job to run a custom Python script to balance and transform the raw data into a format used by BERT model.
# 
# Set the transformation parameters including the instance type, instance count, and train/validation/test split percentages. For the purposes of this lab, you will use a relatively small instance type. Please refer to [this](https://aws.amazon.com/sagemaker/pricing/) link for additional instance types that may work for your use case outside of this lab.
# 
# You can also choose whether you want to balance the dataset or not. In this case, you will balance the dataset to avoid class imbalance in the target variable, `sentiment`. 
# 
# Another important parameter of the model is the `max_seq_length`, which specifies the maximum length of the classified reviews for the RoBERTa model. If the sentence is shorter than the maximum length parameter, it will be padded. In another case, when the sentence is longer, it will be truncated from the right side.
# 
# Since a smaller `max_seq_length` leads to faster training and lower resource utilization, you want to find the smallest power-of-2 that captures `100%` of our reviews.  For this dataset, the `100th` percentile is `115`.  However, it's best to stick with powers-of-2 when using BERT. So let's choose `128` as this is the smallest power-of-2 greater than `115`. You will see below how the shorter sentences will be padded to a maximum length.
# 
# 
# ```
# mean        52.512374
# std         31.387048
# min          1.000000
# 10%         10.000000
# 20%         22.000000
# 30%         32.000000
# 40%         41.000000
# 50%         51.000000
# 60%         61.000000
# 70%         73.000000
# 80%         88.000000
# 90%         97.000000
# 100%       115.000000
# max        115.000000
# ```
# 
# ![](images/distribution_num_words_per_review.png)
# 

# In[9]:


processing_instance_type='ml.c5.xlarge'
processing_instance_count=1
train_split_percentage=0.90
validation_split_percentage=0.05
test_split_percentage=0.05
balance_dataset=True
max_seq_length=128


# To balance and transform our data, you will use a scikit-learn-based processing job. This is essentially a generic Python processing job with scikit-learn pre-installed. You can specify the version of scikit-learn you wish to use. Also pass the SageMaker execution role, processing instance type and instance count.

# In[10]:


from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={'AWS_DEFAULT_REGION': region},                             
    max_runtime_in_seconds=7200
)


# The processing job will be running the Python code from the file `src/prepare_data.py`. In the following exercise you will review the contents of the file and familiarize yourself with main parts of it. 

# <a name='c2w1-ex-2'></a>
# 
# 1. Open the file [src/prepare_data.py](src/prepare_data.py). Go through the comments to understand its content.
# 2. Find and review the `convert_to_bert_input_ids()` function, which contains the RoBERTa `tokenizer` configuration.
# 3. Complete method `encode_plus` of the RoBERTa `tokenizer`. Pass the `max_seq_length` as a value for the argument `max_length`. It defines a pad to a maximum length specified.
# 4. Save the file [src/prepare_data.py](src/prepare_data.py) (with the menu command File -> Save Python File).

# ### _This cell will take approximately 1-2 minutes to run._

# In[13]:


import sys, importlib
sys.path.append('src/')

# import the `prepare_data.py` module
import prepare_data

# reload the module if it has been previously loaded 
if 'prepare_data' in sys.modules:
    importlib.reload(prepare_data)

input_ids = prepare_data.convert_to_bert_input_ids("this product is great!", max_seq_length)
    
updated_correctly = False

if len(input_ids) != max_seq_length:
    print('#######################################################################################################')
    print('Please check that the function \'convert_to_bert_input_ids\' in the file src/prepare_data.py is complete.')
    print('#######################################################################################################')
    raise Exception('Please check that the function \'convert_to_bert_input_ids\' in the file src/prepare_data.py is complete.')
else:
    print('##################')
    print('Updated correctly!')
    print('##################')

    updated_correctly = True


# Review the results of tokenization for the given example (*\"this product is great!\"*):

# In[14]:


input_ids = prepare_data.convert_to_bert_input_ids("this product is great!", max_seq_length)

print(input_ids)
print('Length of the sequence: {}'.format(len(input_ids)))


# Launch the processing job with the custom script passing defined above parameters.

# In[15]:


from sagemaker.processing import ProcessingInput, ProcessingOutput

if (updated_correctly):

    processor.run(code='src/prepare_data.py',
              inputs=[
                    ProcessingInput(source=raw_input_data_s3_uri,
                                    destination='/opt/ml/processing/input/data/',
                                    s3_data_distribution_type='ShardedByS3Key')
              ],
              outputs=[
                    ProcessingOutput(output_name='sentiment-train',
                                     source='/opt/ml/processing/output/sentiment/train',
                                     s3_upload_mode='EndOfJob'),
                    ProcessingOutput(output_name='sentiment-validation',
                                     source='/opt/ml/processing/output/sentiment/validation',
                                     s3_upload_mode='EndOfJob'),
                    ProcessingOutput(output_name='sentiment-test',
                                     source='/opt/ml/processing/output/sentiment/test',
                                     s3_upload_mode='EndOfJob')
              ],
              arguments=['--train-split-percentage', str(train_split_percentage),
                         '--validation-split-percentage', str(validation_split_percentage),
                         '--test-split-percentage', str(test_split_percentage),
                         '--balance-dataset', str(balance_dataset),
                         '--max-seq-length', str(max_seq_length),                         
                         '--feature-store-offline-prefix', str(feature_store_offline_prefix),
                         '--feature-group-name', str(feature_group_name)                         
              ],
              logs=True,
              wait=False)

else:
    print('#######################################')
    print('Please update the code correctly above.')
    print('#######################################')    


# You can see the information about the processing jobs using the `describe` function. The result is in dictionary format. Let's pull the processing job name:

# In[16]:


scikit_processing_job_name = processor.jobs[-1].describe()['ProcessingJobName']

print('Processing job name: {}'.format(scikit_processing_job_name))


# <a name='c2w1-ex-3'></a>
# 
# Pull the processing job status from the processing job description.
# 
# **Instructions**: Print the keys of the processing job description dictionary, choose the one related to the status of the processing job and print the value of it.

# In[17]:


print(processor.jobs[-1].describe().keys())


# In[18]:


scikit_processing_job_status = processor.jobs[-1].describe()['ProcessingJobStatus'] 
print('Processing job status: {}'.format(scikit_processing_job_status))


# Review the created processing job in the AWS console.
# 
# **Instructions**: 
# - open the link
# - notice that you are in the section `Amazon SageMaker` -> `Processing jobs`
# - check the name of the processing job, its status and other available information

# In[19]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/processing-jobs/{}">processing job</a></b>'.format(region, scikit_processing_job_name)))


# Wait for about 5 minutes to review the CloudWatch Logs. You may open the file [src/prepare_data.py](src/prepare_data.py) again and examine the outputs of the code in the CloudWatch logs.

# In[20]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/ProcessingJobs;prefix={};streamFilter=typeLogStreamPrefix">CloudWatch logs</a> after about 5 minutes</b>'.format(region, scikit_processing_job_name)))


# After the completion of the processing job you can also review the output in the S3 bucket.

# In[21]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://s3.console.aws.amazon.com/s3/buckets/{}/{}/?region={}&tab=overview">S3 output data</a> after the processing job has completed</b>'.format(bucket, scikit_processing_job_name, region)))


# Wait for the processing job to complete.
# 
# ### _This cell will take approximately 15 minutes to run._

# In[22]:


get_ipython().run_cell_magic('time', '', '\nrunning_processor = sagemaker.processing.ProcessingJob.from_processing_name(\n    processing_job_name=scikit_processing_job_name,\n    sagemaker_session=sess\n)\n\nrunning_processor.wait(logs=False)')


# _Please wait until ^^ Processing Job ^^ completes above_

# Inspect the transformed and balanced data in the S3 bucket.

# In[23]:


processing_job_description = running_processor.describe()

output_config = processing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'sentiment-train':
        processed_train_data_s3_uri = output['S3Output']['S3Uri']
    if output['OutputName'] == 'sentiment-validation':
        processed_validation_data_s3_uri = output['S3Output']['S3Uri']
    if output['OutputName'] == 'sentiment-test':
        processed_test_data_s3_uri = output['S3Output']['S3Uri']
        
print(processed_train_data_s3_uri)
print(processed_validation_data_s3_uri)
print(processed_test_data_s3_uri)


# In[24]:


get_ipython().system('aws s3 ls $processed_train_data_s3_uri/')


# In[25]:


get_ipython().system('aws s3 ls $processed_validation_data_s3_uri/')


# In[26]:


get_ipython().system('aws s3 ls $processed_test_data_s3_uri/')


# Copy the data into the folder `balanced`.

# In[27]:


get_ipython().system('aws s3 cp $processed_train_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-train/')
get_ipython().system('aws s3 cp $processed_validation_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-validation/')
get_ipython().system('aws s3 cp $processed_test_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-test/')


# Review the training, validation and test data outputs:

# In[28]:


get_ipython().system('head -n 5 ./balanced/sentiment-train/part-algo-1-womens_clothing_ecommerce_reviews.tsv')


# In[29]:


get_ipython().system('head -n 5 ./balanced/sentiment-validation/part-algo-1-womens_clothing_ecommerce_reviews.tsv')


# In[30]:


get_ipython().system('head -n 5 ./balanced/sentiment-test/part-algo-1-womens_clothing_ecommerce_reviews.tsv')


# <a name='c2w1-3.'></a>
# # 3. Query the Feature Store
# In addition to transforming the data and saving in S3 bucket, the processing job populates the feature store with the transformed and balanced data.  Let's query this data using Amazon Athena.

# <a name='c2w1-3.1.'></a>
# ### 3.1. Export training, validation, and test datasets from the Feature Store
# 
# Here you will do the export only for the training dataset, as an example. 
# 
# Use `athena_query()` function to create an Athena query for the defined above Feature Group. Then you can pull the table name of the Amazon Glue Data Catalog table which is auto-generated by Feature Store.

# In[31]:


feature_store_query = feature_group.athena_query()

feature_store_table = feature_store_query.table_name

query_string = """
    SELECT date,
        review_id,
        sentiment, 
        label_id,
        input_ids,
        review_body
    FROM "{}" 
    WHERE split_type='train' 
    LIMIT 5
""".format(feature_store_table)

print('Glue Catalog table name: {}'.format(feature_store_table))
print('Running query: {}'.format(query_string))


# Configure the S3 location for the query results.  This allows us to re-use the query results for future queries if the data has not changed.  We can even share this S3 location between team members to improve query performance for common queries on data that does not change often.

# In[32]:


output_s3_uri = 's3://{}/query_results/{}/'.format(bucket, feature_store_offline_prefix)
print(output_s3_uri)


# <a name='c2w1-ex-4'></a>
# 
# Query the feature store.
# 
# **Instructions**: Use `feature_store_query.run` function passing the constructed above query string and the location of the output S3 bucket.
# 
# ```python
# feature_store_query.run(
#     query_string=..., # query string
#     output_location=... # location of the output S3 bucket
# )
# ```

# In[33]:


feature_store_query.run(
    query_string=query_string, 
    output_location=output_s3_uri 
)

feature_store_query.wait()


# In[34]:


import pandas as pd
pd.set_option("max_colwidth", 100)

df_feature_store = feature_store_query.as_dataframe()
df_feature_store


# Review the Feature Store in SageMaker Studio
# 
# ![](images/sm_studio_extensions_featurestore.png)

# <a name='c2w1-3.2.'></a>
# ### 3.2. Export TSV from Feature Store

# Save the output as a TSV file:

# In[35]:


df_feature_store.to_csv('./feature_store_export.tsv',
                        sep='\t',
                        index=False,
                        header=True)


# In[36]:


get_ipython().system('head -n 5 ./feature_store_export.tsv')


# Upload TSV to the S3 bucket:

# In[37]:


get_ipython().system('aws s3 cp ./feature_store_export.tsv s3://$bucket/feature_store/feature_store_export.tsv')


# Check the file in the S3 bucket:

# In[38]:


get_ipython().system('aws s3 ls --recursive s3://$bucket/feature_store/feature_store_export.tsv')


# <a name='c2w1-3.3.'></a>
# ### 3.3. Check that the dataset in the Feature Store is balanced by sentiment
# 
# Now you can setup an Athena query to check that the stored dataset is balanced by the target class `sentiment`.

# <a name='c2w1-ex-5'></a>
# 
# Write an SQL query to count the total number of the reviews per `sentiment` stored in the Feature Group.
# 
# **Instructions**: Pass the SQL statement of the form 
# 
# ```sql
# SELECT category_column, COUNT(*) AS new_column_name
# FROM table_name
# GROUP BY category_column
# ```
# 
# into the variable `query_string_count_by_sentiment`. Here you would need to use the column `sentiment` and give a name `count_reviews` to the new column with the counts.

# In[39]:


feature_store_query_2 = feature_group.athena_query()


query_string_count_by_sentiment = """
SELECT sentiment, COUNT(*) AS count_reviews
FROM "{}"
GROUP BY sentiment
""".format(feature_store_table)


# <a name='c2w1-ex-6'></a>
# 
# Query the feature store.
# 
# **Instructions**: Use `run` function of the Feature Store query, passing the new query string `query_string_count_by_sentiment`. The output S3 bucket will remain unchanged. You can follow the example above.

# In[40]:


feature_store_query_2.run(
    query_string=query_string_count_by_sentiment, # Replace None
    output_location=output_s3_uri # Replace None
)

feature_store_query_2.wait()

df_count_by_sentiment = feature_store_query_2.as_dataframe()
df_count_by_sentiment


# <a name='c2w1-ex-7'></a>
# 
# Visualize the result of the query in the bar plot, showing the count of the reviews by sentiment value.
# 
# **Instructions**: Pass the resulting data frame `df_count_by_sentiment` into the `barplot` function of the `seaborn` library.
# 
# ```python
# sns.barplot(
#     data=..., 
#     x='...', 
#     y='...',
#     color="blue"
# )
# ```

# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.barplot(
    data=df_count_by_sentiment, 
    x='sentiment', 
    y='count_reviews', 
    color="blue"
)


# Upload the notebook and `prepare_data.py` file into S3 bucket for grading purposes.
# 
# **Note**: you may need to save the file before the upload.

# In[42]:


get_ipython().system('aws s3 cp ./C2_W1_Assignment.ipynb s3://$bucket/C2_W1_Assignment_Learner.ipynb')
get_ipython().system('aws s3 cp ./src/prepare_data.py s3://$bucket/src/C2_W1_prepare_data_Learner.py')


# Please go to the main lab window and click on `Submit` button (see the `Finish the lab` section of the instructions).

# In[ ]:




